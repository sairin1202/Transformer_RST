import torch
from mask import nopeak_mask, create_masks
import torch.nn.functional as F
import math
import numpy as np

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def init_vars(src, root, mask_tok, transformer, eventer, SRC, TRG, beam_size, max_len):

    init_tok = TRG('<sos>')
    outputs = torch.LongTensor([[init_tok]])
    outputs = outputs.cuda()

    trg_mask = nopeak_mask(1)
    trg_mask = trg_mask.cuda()

##################################################
    src_lens_flat = [len(s) for s in src]
    src_toks = torch.zeros(1, len(src), max(src_lens_flat)).long()
    for i, s in enumerate(src):
        end = src_lens_flat[i]
        src_toks[0, i, :end] = torch.LongTensor(s[:end])
    src = src_toks
    src = src.cuda()
    sent_num = src.size(1)

    mask_tok = torch.LongTensor(mask_tok).unsqueeze(0)
    # print(mask_tok.size())
    mask_tok = tile(mask_tok, 1, max(src_lens_flat))
    mask_tok = tile(mask_tok, 2, max(src_lens_flat))
    mask_tok = mask_tok.cuda()
    # B * S
    src_masks = [None]*sent_num
    for i in range(sent_num):
        src_masks[i], _ = create_masks(src[:, i], None)
    for i in range(sent_num):
        src_masks[i] = src_masks[i].squeeze(1).cuda()
    src_masks = torch.stack([m for m in src_masks], dim=1)
    src_word_masks = src_masks.view(src_masks.size(0), 1, -1)
    #  print("src_word_masks", src_word_masks.size())
    
    events = [None]*sent_num
    for i in range(sent_num):
        events[i] = transformer.encoder(src[:, i], src_masks[:,i].unsqueeze(1))

    eventers = torch.cat([e for e in events], dim=-2)
    eventers = eventer(eventers, mask_tok)
    eventers = eventers.view(eventers.size(0), sent_num, -1, eventers.size(2))
    # print(eventers.size())
    root = torch.LongTensor([root]).cuda()
    feat_root = root.view(-1,1,1,1).expand(-1,1,eventers.size(2), eventers.size(3))
    root_feat = torch.gather(eventers, 1, feat_root).squeeze(1)
    mask_root = root.view(-1,1,1).expand(-1,1,src_masks.size(2))
    root_masks = torch.gather(src_masks, 1, mask_root)
    # print(root_feat.size(), root_masks.size())
    # pred = transformer.out(transformer.decoder(trg_input, sent_feats, mask_sent.unsqueeze(1), trg_mask)[0])
    # print(root_masks.data.cpu().numpy())
    pred = transformer.out(transformer.decoder(outputs, root_feat, root_masks, trg_mask)[0])
    #
    out = F.softmax(pred, dim=-1)

    probs, ix = out[:, -1].data.topk(beam_size)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(beam_size, max_len).long()
    outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(beam_size, eventers.size(-2), eventers.size(-1))
    e_outputs = e_outputs.cuda()
    e_outputs[:, :] = root_feat

    return outputs, e_outputs, log_scores, root_masks

def k_best_outputs(outputs, out, log_scores, i, k):
    # print("outputs", outputs.size())
    probs, ix = out[:, -1].data.topk(k)
    #  print("out", out.size())
    #  print("probs", probs.size())
    #  print("ix", ix.size())
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores

def beam_search(src, root, mask_sent, transformer, eventer, SRC, TRG, beam_size, max_len=99):
    outputs, e_outputs, log_scores, mask = init_vars(src, root, mask_sent, transformer, eventer, SRC, TRG, beam_size, max_len)
    eos_tok = TRG('<eos>')
    ind = None
    for i in range(2, max_len):

        trg_mask = nopeak_mask(i)
        #  print("output",outputs.size())
        out = transformer.out(transformer.decoder(outputs[:,:i],
        e_outputs, mask, trg_mask)[0])

        out = F.softmax(out, dim=-1)
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, beam_size)

        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])
        if num_finished_sentences == beam_size:
            alpha = 0.7
            # div = 1/((outputs==eos_tok).nonzero()[:,1].type_as(log_scores)**alpha)
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    if ind is None:
        print("Invalid output")
        return ' '
        length = (outputs[0]==eos_tok).nonzero()[0]
        return ' '.join([TRG.idx2word[tok.item()] for tok in outputs[0][1:length]])

    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([TRG.idx2word[tok.item()] for tok in outputs[ind][1:length]])
