import math
from tqdm import tqdm
import argparse
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from mask import create_masks, nopeak_mask
from beam import beam_search
import pickle
import argparse
from model import get_model
from nltk.corpus import wordnet
from build_vocab import Vocabulary
from copy import deepcopy
from dataset import get_loader
import matplotlib.pyplot as plt

def generate_sentence(sentence, root, transformer, eventer, SRC, TRG, beam_size):

    transformer.eval()
    eventer.eval()
    src_idxs = []
    masks = []
    for src, mask in sentence.items():
        
        list_mask = []
        for m in mask:
            if m == "0":
                list_mask.append(0)
            else:
                list_mask.append(1)
        masks.append(list_mask[:])

        src_idx = []
        src = src.split(" ")
        src_idx.append(SRC('<sos>'))
        for tok in src:
            src_idx.append(SRC(tok))
        src_idx.append(SRC('<eos>'))
        src_idxs.append(deepcopy(src_idx))

    # np.random.shuffle(src_idxs)
    src = src_idxs
    hypo = beam_search(src, root, masks, transformer, eventer, SRC, TRG, beam_size)
    return hypo

def generate(transformer, eventer, SRC, TRG, beam_size):
    transformer.eval()
    eventer.eval()
    with open('data/test_sent_dic.pkl', 'rb') as f:
        sentences = pickle.load(f)
    with open('data/test_trg.txt', 'r') as f:
        test_data = f.readlines()
    with open('data/test_root.pkl','rb') as f:
        test_root = pickle.load(f)
    tests = []
    hypos = []
    num = 0 
    for ids in tqdm(range(100)):
        if ids not in sentences:
            continue
        story = sentences[ids] 
        # tests.append(test_data[ids])
        root = test_root[ids]
        num += 1
        hypo = generate_sentence(story, root, transformer, eventer, SRC, TRG, beam_size)
        hypos.append(hypo+'\n')
    with open("res/hypo.txt", "w") as f:
        f.writelines(hypos)
    # with open("res/test.txt", "w") as f:
    #     f.writelines(tests)

def eval(transformer, eventer, dataset):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    transformer.eval()
    eventer.eval()
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
            src_masks[i] = src_masks[i].squeeze(1).cuda()
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
        # eventers = transformer.encoder.norm(eventers)
        # print(eventers.size())
        # print(eventers.size())
        # sent_feats = eventers
        eventers = eventer(eventers, mask_tok)
        eventers = eventers.view(eventers.size(0), sent_num, -1, eventers.size(2))
        feat_root = root.view(-1,1,1,1).expand(-1,1,eventers.size(2), eventers.size(3))
        root_feat = torch.gather(eventers, 1, feat_root).squeeze(1)
        mask_root = root.view(-1,1,1).expand(-1,1,src_masks.size(2))
        root_masks = torch.gather(src_masks, 1, mask_root)
        # print(root_feat.size(), root_masks.size())
        # pred = transformer.out(transformer.decoder(trg_input, sent_feats, mask_sent.unsqueeze(1), trg_mask)[0])
        # print(root_masks.data.cpu().numpy())
        pred = transformer.out(transformer.decoder(trg_input, root_feat, root_masks, trg_mask)[0])
        ys = trgs[:, 1:].contiguous().view(-1).cuda()

        loss = criterion(pred.view(-1, pred.size(-1)), ys)
        # print(loss)
        total_loss.append(loss.item())
    return np.mean(total_loss)

def shuffle_eval(transformer, eventer, dataset):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    transformer.eval()
    eventer.eval()
    total_loss = []
    for index, (srcs, srcs_len, trgs, trgs_len, mask_tok, root) in enumerate(tqdm(dataset)):
        sent_num = srcs.size(1)
        srcs = srcs.cuda()
        srcs = srcs[:,torch.randperm(srcs.size(1)),:]
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
            src_masks[i] = src_masks[i].squeeze(1).cuda()
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
        # eventers = transformer.encoder.norm(eventers)
        # print(eventers.size())
        # print(eventers.size())
        # sent_feats = eventers
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

        loss = criterion(pred.view(-1, pred.size(-1)), ys)
        # print(loss)
        total_loss.append(loss.item())
    return np.mean(total_loss)


def get_attn(transformer, eventer, pool, pe, dataset):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    transformer.eval()
    eventer.eval()
    total_loss = []
    for idx, (srcs, srcs_len, trgs, trgs_len, mask_tok, mask_sent) in enumerate(tqdm(dataset)):
        sent_num = srcs.size(1)
        srcs = srcs.cuda()
        mask_tok = mask_tok.cuda()
        mask_sent = mask_sent.cuda()
        # B * S
        trgs = trgs.cuda()
        trg_input = trgs[:, :-1]
        trg_input = trg_input.cuda()
        src_masks = [None]*sent_num
        trg_mask = None




        for i in range(sent_num):
            src_masks[i], trg_mask = create_masks(srcs[:, i], trg_input)
        for i in range(sent_num):
            src_masks[i] = src_masks[i].squeeze(1).cuda()
        src_masks = torch.stack([m for m in src_masks], dim=1)
        src_word_masks = src_masks.view(src_masks.size(0), 1, -1)
        trg_mask = trg_mask.cuda()


        events = [None]*sent_num
        for i in range(sent_num):
            events[i] = transformer.encoder(srcs[:, i], src_masks[:, i].unsqueeze(1))
            # print(events[i].size(), src_masks.size())
            events[i] = pool(events[i], src_masks[:, i])

        eventers = torch.stack([e for e in events], dim=1)
        eventers = pe(eventers)

        sent_feats = eventers
        out, attn1 = transformer.decoder(trg_input, sent_feats, mask_sent.unsqueeze(1), trg_mask)

        r=torch.randperm(srcs.size(1))
        srcs=srcs[:, r, :]
        recover = {}
        for x in range(len(r)):
            recover[r[x].item()] = x
        # print(r)
        new_r = []
        for x in range(len(r)):
            new_r.append(recover[x])

        src_masks = [None]*sent_num
        for i in range(sent_num):
            src_masks[i], trg_mask = create_masks(srcs[:, i], trg_input)
        for i in range(sent_num):
            src_masks[i] = src_masks[i].squeeze(1).cuda()
        src_masks = torch.stack([m for m in src_masks], dim=1)
        src_word_masks = src_masks.view(src_masks.size(0), 1, -1)
        trg_mask = trg_mask.cuda()


        events = [None]*sent_num
        for i in range(sent_num):
            events[i] = transformer.encoder(srcs[:, i], src_masks[:, i].unsqueeze(1))
            events[i] = pool(events[i], src_masks[:, i])

        eventers = torch.stack([e for e in events], dim=1)
        eventers = pe(eventers)
        sent_feats = eventers
        out, attn2 = transformer.decoder(trg_input, sent_feats, mask_sent.unsqueeze(1), trg_mask)
        attn1 = np.array(attn1[-1])
        attn2 = np.array(attn2[-1])

        attn1 = attn1.mean(axis=0).squeeze()
        attn2 = attn2.mean(axis=0).squeeze()
        attn2 = attn2[:,new_r]
        print(attn1.shape, attn2.shape)
        for k in range(len(attn2)):
            plt.figure(figsize=(10,5))
            plt.ylim(0,1)
            # print(attn.shape, np.array(word_sims).shape)
            x = [k for k in range(len(attn1[0]))]
            y = attn1[k]
            # print(y)
            plt.plot(x, y)
            x = [k for k in range(len(attn2[0]))]
            y = attn2[k]

            plt.plot(x, y)

            plt.ylabel('word')
            plt.xlabel('sentence')
            plt.savefig(f'attn_order/attn{k}.png')
            plt.clf()
        if idx >2:
            break

    return 


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-beam_size', type=int, default=4)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=300)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-src_lang', default='en')
    parser.add_argument('-trg_lang', default='en')
    parser.add_argument('-heads', type=int, default=4)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-load_epoch', type=int)

    opt = parser.parse_args()
    assert opt.beam_size > 0
    assert opt.max_len > 10

    # load vocab
    with open("vocab/src_vocab.pkl", 'rb') as f:
        SRC = pickle.load(f)
    with open("vocab/trg_vocab.pkl", 'rb') as f:
        TRG = pickle.load(f)
    # opt.load_epoch = None
    transformer, eventer = get_model(SRC, TRG, opt.load_epoch, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    test_dataset = get_loader("data/test_sent_dic.pkl", "data/test_trg.txt", "data/test_root.pkl", SRC, TRG, 1)
    #  evaluate
    loss1 = eval(transformer, eventer, test_dataset)
    # print(loss1)
    loss2 = shuffle_eval(transformer, eventer, test_dataset)
    print("Loss: ", loss1, loss2)

    # generate
    #  greedy_generate(model, SRC, TRG)
    generate(transformer, eventer, SRC, TRG, opt.beam_size)


    # get_attn(transformer, eventer, pool, pe, test_dataset)

if __name__ == '__main__':
    main()
