import argparse
import time
import torch
import torch.nn as nn
from model import get_model
import torch.nn.functional as F
from mask import create_masks
import dill as pickle
import itertools
from tqdm import tqdm
from build_vocab import Vocabulary
from dataset import get_loader
import math
from evaluation import eval, shuffle_eval
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)
torch.cuda.set_device(0)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def random_seeding(seed_value=1202, use_cuda=True):
    numpy.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    if use_cuda: torch.cuda.manual_seed_all(seed_value) # gpu vars


def train_model(transformer, eventer, dataset, test_dataset, epochs, criterion, optimizer, SRC, TRG):

    print("training model...")
    # eval_loss1 = eval(transformer, eventer, test_dataset)
    # eval_loss2 = shuffle_eval(transformer, eventer, test_dataset)
    for epoch in range(epochs):
        transformer.train()
        eventer.train()
        cur_lr = get_lr(optimizer)
        print("Current lr ", cur_lr)
        total_loss = []
        for index, (srcs, srcs_len, trgs, trgs_len, mask_tok, root) in enumerate(tqdm(dataset)):
            sent_num = srcs.size(1)
            srcs = srcs.cuda()
            mask_tok = mask_tok.cuda()
            root = root.cuda()
            # B * S
            trgs = trgs.cuda()
            trg_input = trgs[:, :-1]
            trg_input = trg_input.cuda()
            src_masks = [None]*sent_num
            trg_mask = None
            for i in range(sent_num):
                src_masks[i], trg_mask = create_masks(srcs[:, i], trg_input)
            for i in range(sent_num):
                src_masks[i] = src_masks[i].squeeze().cuda()
            src_masks = torch.stack([m for m in src_masks], dim=1)
            src_word_masks = src_masks.view(src_masks.size(0), 1, -1)
            # print(src_word_masks.size())
            src_word_tok_masks = src_word_masks.repeat(1,src_word_masks.size(2),1)
            # print("word_mask", src_word_tok_masks[0][0])
            # print("mask_tok", mask_tok[0][0])
            mask_tok = mask_tok*src_word_tok_masks.long()
            # print("mask_tok", mask_tok[0][0])
            trg_mask = trg_mask.cuda()

            events = [None]*sent_num
            for i in range(sent_num):
                events[i] = transformer.encoder(srcs[:, i], src_masks[:, i].unsqueeze(1))
                # print(events[i].size())
                # print(src_masks[0,i])
                # events[i] = pool(events[i], src_masks[:, i])

            eventers = torch.cat([e for e in events], dim=-2)
            eventers = eventer(eventers, mask_tok)
            eventers = eventers.view(eventers.size(0), sent_num, -1, eventers.size(2))

            feat_root = root.view(-1,1,1,1).expand(-1,1,eventers.size(2), eventers.size(3))
            root_feat = torch.gather(eventers, 1, feat_root).squeeze(1)
            mask_root = root.view(-1,1,1).expand(-1,1,src_masks.size(2))
            root_masks = torch.gather(src_masks, 1, mask_root)
            # print(root_feat.size(), root_masks.size())
            # pred = transformer.out(transformer.decoder(trg_input, sent_feats, mask_sent.unsqueeze(1), trg_mask)[0])
            pred = transformer.out(transformer.decoder(trg_input, root_feat, root_masks, trg_mask)[0])
            ys = trgs[:, 1:].contiguous().view(-1).cuda()

            optimizer.zero_grad()
            loss = criterion(pred.view(-1, pred.size(-1)), ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(eventer.parameters(), 0.1)
            optimizer.step()
            total_loss.append(loss.item())
        print(f"Epoch {epoch} training loss : ", sum(total_loss)/len(total_loss))
        eval_loss1 = eval(transformer, eventer, test_dataset)
        eval_loss2 = shuffle_eval(transformer, eventer, test_dataset)
        print(f"Epoch {epoch} evaluation loss : ", eval_loss1, eval_loss2)
        torch.save(transformer.state_dict(), f'models/transformer{epoch}.pth')
        torch.save(eventer.state_dict(), f'models/eventer{epoch}.pth')

def main():
    random_seeding()
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_data', type=str, default='data/sent_dic.pkl')
    parser.add_argument('-trg_data', type=str, default='data/trg.txt')
    parser.add_argument('-test_src_data', type=str, default='data/test_sent_dic.pkl')
    parser.add_argument('-test_trg_data', type=str, default='data/test_trg.txt')
    parser.add_argument('-root_data', type=str, default='data/root.pkl')
    parser.add_argument('-test_root_data', type=str, default='data/test_root.pkl')
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-d_model', type=int, default=300)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-heads', type=int, default=4)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=64)
    parser.add_argument('-lr', type=int, default=0.001)
    parser.add_argument('-max_strlen', type=int, default=120)

    opt = parser.parse_args()
    # load vocab
    with open('./vocab/src_vocab.pkl', 'rb') as f:
        SRC = pickle.load(f)
    with open('./vocab/trg_vocab.pkl', 'rb') as f:
        TRG = pickle.load(f)
    # get model
    transformer, eventer = get_model(SRC, TRG, None, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print("transformer trainable parameters: ", total_params)
    total_params = sum(p.numel() for p in eventer.parameters() if p.requires_grad)
    print("eventer trainable parameters: ", total_params)
    optimizer = torch.optim.Adam(itertools.chain(transformer.parameters(), eventer.parameters()), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    dataset = get_loader(opt.src_data, opt.trg_data, opt.root_data, SRC, TRG, opt.batchsize)
    test_dataset = get_loader(opt.test_src_data, opt.test_trg_data, opt.test_root_data, SRC, TRG, 1, False)
    train_model(transformer, eventer, dataset, test_dataset, opt.epochs, criterion, optimizer, SRC, TRG)


if __name__ == "__main__":
    main()
