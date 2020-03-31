import numpy as np
import os
from torch.utils.data import Dataset
import glob
import numpy as np
from tqdm import tqdm
import pickle
import torch
from copy import deepcopy

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

class ROCdataset():
    def __init__(self, src_dir, trg_dir, root_dir, src_vocab, trg_vocab):
        print("Loading data ...  ")
        # loading training data
        with open(src_dir, 'rb') as f:
            self.src = pickle.load(f) 
        
        with open(trg_dir, 'r') as f:
            self.trg = f.readlines()
        self.trg = [t[:-1].lower() for t in self.trg]


        with open(root_dir, 'rb') as f:
            self.root = pickle.load(f) 

        print('src data length', len(self.src))
        print('trg data length', len(self.trg))
        self.data_len = len(self.trg)

        print("Loading vocab ... ")
        # loading vocab data
        self.SRC = src_vocab
        self.TRG = trg_vocab


    def __getitem__(self, index):
        src_idxs = []
        if index not in self.src:
            index = index + 1 
        srcs = self.src[index]
        masks = []
        for src, mask in srcs.items():
            masks.append(mask) 
            src_idx = []
            src = src.split(" ")
            src_idx.append(self.SRC('<sos>'))
            for tok in src:
                src_idx.append(self.SRC(tok))
            src_idx.append(self.SRC('<eos>'))
            src_idxs.append(deepcopy(src_idx))
        

        trg_idx = []
        trg = self.trg[index]
        #  print(trg)
        trg = trg.split(" ")
        trg_idx.append(self.TRG('<sos>'))
        for tok in trg:
            trg_idx.append(self.TRG(tok))
        trg_idx.append(self.TRG('<eos>'))
        
        
        src = src_idxs
        trg = trg_idx


        root = self.root[index]
        return src, trg, masks, root

    def __len__(self):
        return self.data_len


def collate_fn(data):
    #  data.sort(key=lambda x: len(x[0]), reverse=True)
    srcss, trgss, masks, root = zip(*data)
    root = torch.LongTensor(root)
    sent_nums = [len(s) for s in srcss]
    src_lens = [[len(s) for s in src] for src in srcss]
    trg_lens = [len(trg) for trg in trgss]
    src_lens_flat = [len(s) for src in srcss for s in src] 
    src_toks = torch.zeros(len(srcss), max(sent_nums), max(src_lens_flat)).long()
    for i, srcs in enumerate(srcss):
        for j, src in enumerate(srcs):
            end = src_lens[i][j]
            src_toks[i, j, :end] = torch.LongTensor(src[:end])
    trg_toks = torch.zeros(len(trgss), max(trg_lens)).long()
    for i, trgs in enumerate(trgss):
        end = trg_lens[i]
        trg_toks[i, :end] = torch.LongTensor(trgs[:end])
    #  for mask in masks:
        #  for m in mask:
            #  assert len(m)<=max(sent_nums), str(len(m))+">"+str(max(sent_nums))

    mask_toks = torch.zeros(len(srcss), max(sent_nums), max(sent_nums)).long()
    for i1, m in enumerate(masks):
        for i2, n in enumerate(m):
            for i3, l in enumerate(n):
                # if i2 == i3:
                #     mask_toks[i1,i2,i3] = 1
                if l == "0":
                    mask_toks[i1,i2,i3] = 0
                else:
                    mask_toks[i1,i2,i3] = 1
    mask_toks = tile(mask_toks, 1, max(src_lens_flat))
    mask_toks = tile(mask_toks, 2, max(src_lens_flat))

    return src_toks, src_lens, trg_toks, trg_lens, mask_toks, root


def get_loader(src_dir, trg_dir, root_dir, src_vocab, trg_vocab, batch_size, shuffle=True):
    dataset = ROCdataset(src_dir, trg_dir, root_dir, src_vocab, trg_vocab)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=4,
                                              collate_fn = collate_fn,
                                              drop_last=True)
    return data_loader
