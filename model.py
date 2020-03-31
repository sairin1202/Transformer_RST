import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
import pickle
import numpy as np


use_glove = True



# Embedder
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model, weights_matrix):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        if weights_matrix is not None:
            print("Initialize embedding layer with Glove ", weights_matrix.shape)
            self.embed.load_state_dict({'weight': torch.Tensor(weights_matrix)})
            self.embed.weight.requires_grad = False
    def forward(self, x):
        # print(self.embed.weight.grad)
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:,:seq_len]
        pe.require_grad = False
        pe = pe.cuda()
        x = x + pe
        # print(pe.size())
        return self.dropout(x)

class SentPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = self.pe[:,:seq_len]
        pe.require_grad = False
        pe = pe.cuda()
        x = x + pe
        # print(pe.size())
        return self.dropout(x)


# SubLayers
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)
    # if mask is not None:
    #     scores = scores.masked_fill(mask == 0, 1e-18)
    attn = scores
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)


        # calculate attention using function we will define next
        scores, attn = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        # B * H * S * dim/H
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        # B * S * dim
        output = self.out(concat)
        # B * S * dim
        return output, attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask)[0])
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask)[0])
        x2 = self.norm_2(x)
        attn2, weights = self.attn_2(x2, e_outputs, e_outputs, src_mask)
        x = x + self.dropout_2(attn2)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, weights



# Transformer
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, weights_matrix):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model, weights_matrix)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        # print("embeded src size", x.size())
        x = self.pe(x)
        # print("encoder1 size", x.size())
        for i in range(self.N):
            x = self.layers[i](x, mask)
        # print("encoder2 size", x.size())
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, weights_matrix):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model, weights_matrix)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        decoder_attns = []
        for i in range(self.N):
            x, attn  = self.layers[i](x, e_outputs, src_mask, trg_mask)
            attn = attn.squeeze()
            decoder_attns.append(attn.tolist())
        return self.norm(x), decoder_attns


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, src_weights_matrix=None, trg_weights_matrix=None):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout, src_weights_matrix)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout, trg_weights_matrix)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):

        e_outputs = self.encoder(src, src_mask)
        # B * S * dim
        d_output, attns = self.decoder(trg, e_outputs, src_mask, trg_mask)
        # B * S * dim
        output = self.out(d_output)
        return output, attns




class EventerLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        # self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x,x,mask)[0])
        # x2 = self.norm_2(x)
        # x = x + self.dropout_2(self.ff(x2))
        return x

class Eventer(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EventerLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, x, mask):
        # print("encoder1 size", x.size())
        for i in range(self.N):
            #  print(mask[0])
            x = self.layers[i](x, mask)
        # print("encoder2 size", x.size())
        return self.norm(x)



def get_model(src_vocab, trg_vocab, load_epoch, d_model, n_layers, heads, dropout):

    assert d_model % heads == 0
    assert dropout < 1
    # init glove vector
    src_weights_matrix = None
    trg_weights_matrix = None
    if use_glove:
        with open('glove.6B/glove6B.300vector.pkl', 'rb') as f:
            glove = pickle.load(f)

        matrix_len = len(src_vocab)
        src_weights_matrix = np.zeros((matrix_len, 300))
        words_found = 0
        words_unfound = 0
        for i, word in src_vocab.idx2word.items():
            try:
                src_weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                words_unfound += 1
                src_weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))
        print("glove found src vocab size", words_found, "unfound size", words_unfound)

        matrix_len = len(trg_vocab)
        trg_weights_matrix = np.zeros((matrix_len, 300))
        words_found = 0
        words_unfound = 0
        for i, word in trg_vocab.idx2word.items():
            try:
                trg_weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                words_unfound += 1
                trg_weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))
        print("glove found trg vocab size", words_found, "unfound size", words_unfound)
    #  init model
    transformer = Transformer(len(src_vocab), len(trg_vocab), d_model, n_layers, heads, dropout, src_weights_matrix, trg_weights_matrix)
    eventer = Eventer(d_model, n_layers, heads, dropout)
    if load_epoch is not None:
        print("loading pretrained weights...")
        transformer.load_state_dict(torch.load(f'./models/transformer{load_epoch}.pth'))
        eventer.load_state_dict(torch.load(f'./models/eventer{load_epoch}.pth'))
    else:
        total_cnt = 0
        cnt = 0
        for p in transformer.parameters():
            total_cnt += p.view(-1).size(0)
            if p.dim() > 1 and p.requires_grad == True:
                cnt += p.view(-1).size(0)
                nn.init.xavier_uniform_(p)
        for p in eventer.parameters():
            total_cnt += p.view(-1).size(0)
            if p.dim() > 1 and p.requires_grad == True:
                cnt += p.view(-1).size(0)
                nn.init.xavier_uniform_(p)
        print("Total parameters {} , initilized {} parameters".format(total_cnt, cnt))

    transformer = transformer.cuda()
    eventer = eventer.cuda()
    return transformer, eventer

