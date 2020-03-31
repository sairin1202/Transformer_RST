import pickle
from copy import deepcopy
import glob
from tqdm import tqdm
from collections import OrderedDict
def parse(file):
    with open(file,'r') as f:
        data = f.readlines()
    strings = []
    for s in data:
        strings.extend(s)
    
    sent_start = False
    sentences = []
    sentence = []
    signals = []
    count = 0
    sent_count = 0
    match = {}
    for i, c in enumerate(strings):
        if c == "[" and "".join(strings[i:i+6]) == "[N][N]":
           signals.append("NN") 
        if c == "[" and "".join(strings[i:i+6]) == "[S][N]":
           signals.append("SN")
        if c == "[" and "".join(strings[i:i+6]) == "[N][S]":
           signals.append("NS")
        if c == "[" and "".join(strings[i:i+6]) == "[S][S]":
           signals.append("SS") 
        if sent_start:
            sentence.append(c)
        if c == "!" and strings[i-1] == "_":
            sent_start = True
        if c == "_" and strings[i-1] == "!":
            sent_start = False
            if "".join(sentence)[-6:] == " <s>!_":
                match[sent_count] = count
                count += 1
                sent_count += 1
                sentences.append("".join(sentence[:-6]))
            else:
                match[sent_count] = count
                sent_count += 1
                sentences.append("".join(sentence[:-2]))
            sentence = []
        if c == ")" and sent_start == False:
            signal = signals.pop()
            sentences.append("pop"+signal)
    #  print(sentences)
    if len(sentence):
        sentences.append("".join(sentence[:]))
        print("need check")
    #  print(sentences)
    sent_stack = []
    sent_dict = OrderedDict()
    sent_num = 0
    for sent in sentences:
        if not (sent[:3] == "pop" and len(sent) == 5):
            sent_num += 1
            sent_dict[sent] = -1
    #  print(sent_num)
    assert sent_num == sent_count
    cnt = 0
    sent_set = set()
    for sent in sentences:
        if not (sent[:3] == "pop" and len(sent) == 5):
            while sent in sent_set:
                sent += " #"
        sent_set.add(sent)
        #  print(sent)
        if not (sent[:3] == "pop" and len(sent) == 5):
            mask = ["0"]*sent_num
            #  print(mask, cnt)
            mask[cnt] = "1"
            cnt += 1
            sent_stack.append((sent, mask, mask))
        else:
            sign = sent[3:]
            sent1, mask1, raw_mask1 = sent_stack.pop()
            sent2, mask2, raw_mask2 = sent_stack.pop()
            if sign == "NS" or sign == "NN":
                mask2 = addmask(raw_mask1, mask2)
                sent_stack.append((sent2, mask2, raw_mask2))
                sent_dict[sent1] = "".join(mask1)
            if sign == "SS":
                print("occur SS")
            if sign == "SN":
                mask1 = addmask(raw_mask2, mask1)
                sent_stack.append((sent1, mask1, raw_mask1))
                sent_dict[sent2] = "".join(mask2) 
    for (sent, mask, raw_mask) in sent_stack:
        sent_dict[sent] = "".join(mask)
    #  print(sent_dict)
    return sent_dict, match

def addmask(a, b):
    res = []
    for i in range(len(a)):
        if a[i] != b[i]:
            res.append("1")
        else:
            res.append("0")
        if a[i] == "1" and b[i] == "1":
            print("wrong mask")
    return res

def mergemask(a, b):
    res = []
    for i in range(len(a)):
        if a[i] == 1 or b[i] == 1:
            res.append(1)
        else:
            res.append(0)
    return res



if __name__ == "__main__":
    #  parse("data/tree/src/90530.txt.tree")
    files = glob.glob("data/tree/test_src/*.tree")
    res_dict = {}
    for f in tqdm(files):
        #  try:
        num = int(f.split("/")[-1].split(".")[0])
        dic , match = parse(f)
        sents = {}
        masks = {}
        cnt = 0
        for sent, mask in dic.items():
            if match[cnt] not in sents:
                sents[match[cnt]] = sent
            else:
                sents[match[cnt]] += " "+sent
            temp_mask = [0,0,0,0]
            for x, m in enumerate(mask):
                if m == '1':
                    temp_mask[match[x]] = 1
            mask = temp_mask
            if match[cnt] not in masks:
                masks[match[cnt]] = mask
            else:
                masks[match[cnt]] = mergemask(mask, masks[match[cnt]])
            cnt += 1
        #print(masks)
        #print(sents) 
        res = OrderedDict()
        for i in range(4):
            res[sents[i]] = "".join([str(m) for m in masks[i]])
        res_dict[num] = deepcopy(res)
        #  except:
            #  print(f)
    with open('data/test_sent_dic.pkl','wb') as f:
        pickle.dump(res_dict, f)

