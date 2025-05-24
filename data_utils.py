from torch.utils.data import Dataset, DataLoader
import os
import json
import torch
from transformers import BartTokenizer, PegasusTokenizer
from functools import partial
import math


def to_cuda(batch, gpuid):
    for n in batch:
        if batch[n] is None:
            print(n)
        if n not in ["data", "abstract_tag", "article_tag"]:
            batch[n] = batch[n].to(gpuid)


class BrioDataset(Dataset):
    def __init__(
        self, 
        fdir, 
        model_type, 
        max_len=-1, 
        is_test=False, 
        total_len=512, 
        is_sorted=True, 
        max_num=-1, 
        is_untok=True, 
        is_pegasus=False, 
        num=-1,
        sentiment='POSITIVE',
        use_sent=False,
        use_cls=False
    ):
        """ data format: article, abstract, [(candidiate_i, score_i)] """
        self.isdir = os.path.isdir(fdir)
        if self.isdir:
            self.fdir = fdir
            if num > 0:
                self.num = min(len(os.listdir(fdir)), num)
            else:
                self.num = len(os.listdir(fdir))
        else:
            with open(fdir) as f:
                self.files = [x.strip() for x in f]
            if num > 0:
                self.num = min(len(self.files), num)
            else:
                self.num = len(self.files)
        if is_pegasus:
            self.tok = PegasusTokenizer.from_pretrained(model_type, verbose=False)
        else:
            self.tok = BartTokenizer.from_pretrained(model_type, verbose=False)
        self.maxlen = max_len
        self.is_test = is_test
        self.total_len = total_len
        self.sorted = is_sorted
        self.maxnum = max_num
        self.is_untok = is_untok
        self.is_pegasus = is_pegasus
        self.sent_type = sentiment
        self.use_sent = use_sent
        self.use_cls = use_cls
        self.ptm_path = "valhalla/bart-large-sst2"
        if self.use_sent:
            self.tok_classifier = BartTokenizer.from_pretrained(self.ptm_path)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        if self.isdir:
            with open(os.path.join(self.fdir, "%d.json"%idx), "r") as f:
                data = json.load(f)
        else:
            with open(self.files[idx]) as f:
                data = json.load(f)
        if self.use_sent and self.use_cls:
            sent_prompt = self.sent_type + 'summary'
        else:
            sent_prompt = ''
        if self.is_untok:
            article = data["article_untok"]
        else:
            article = data["article"]
        src_txt = " ".join(article) + sent_prompt# 这里将所有句子整合成了一个大句子
        src = self.tok.batch_encode_plus([src_txt], max_length=self.total_len, return_tensors="pt", pad_to_max_length=False, truncation=True)
        src_input_ids = src["input_ids"]
        src_input_ids = src_input_ids.squeeze(0)
        src_attention_mask = src["attention_mask"]
        src_attention_mask = src_attention_mask.squeeze(0)
        if self.is_untok:
            abstract = data["abstract_untok"]
        else:
            abstract = data["abstract"]
        if self.maxnum > 0:
            candidates = data["candidates_untok"][:self.maxnum]
            _candidates = data["candidates"][:self.maxnum]
            data["candidates"] = _candidates
        if self.sorted:
            candidates = sorted(candidates, key=lambda x:x[1], reverse=True)
            _candidates = sorted(_candidates, key=lambda x:x[1], reverse=True)
            data["candidates"] = _candidates
        if not self.is_untok:
            candidates = _candidates
        cand_txt = [" ".join(abstract)] + [" ".join(x[0]) for x in candidates]
        cand = self.tok.batch_encode_plus(cand_txt, max_length=self.maxlen, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True)
        candidate_ids = cand["input_ids"]
        if self.use_sent:
            # 因为参与损失函数计算，采用与cand相似的逻辑
            # 为了让长度匹配，等长切分作为候选
            src_sent_txt = get_sent_txt(article, data["article_tag"], self.sent_type)
            cand_sent_txt = [src_sent_txt]
            # print("cand_sent_txt", end="")
            # print(cand_sent_txt)
            src_sent = self.tok.batch_encode_plus(cand_sent_txt, max_length=self.maxlen, return_tensors="pt", pad_to_max_length=False, truncation=True, padding=True)
            src_sent_input_ids = src_sent["input_ids"]
            # print_var('src_sent_input_ids', src_sent_input_ids, 'after batch encode')
            # print("src_sent_ids", end="")
            # print(src_sent_input_ids)
            src_tag = data["article_tag"]
            abs_tag = data["abstract_tag"] if "abstract_tag" in data.keys() else None
        else:
            src_sent_input_ids = None
            src_tag = None
            abs_tag = None
        if self.is_pegasus:
            # add start token
            _candidate_ids = candidate_ids.new_zeros(candidate_ids.size(0), candidate_ids.size(1) + 1)
            _candidate_ids[:, 1:] = candidate_ids.clone()
            _candidate_ids[:, 0] = self.tok.pad_token_id
            candidate_ids = _candidate_ids
            if self.use_sent:
                _src_sent_input_ids = src_sent_input_ids.new_zeros(src_sent_input_ids.size(0), src_sent_input_ids.size(1)+1)
                _src_sent_input_ids[:, 1:] = src_sent_input_ids.clone()
                _src_sent_input_ids[:, 0] = self.tok.pad_token_id
                src_sent_input_ids = _src_sent_input_ids
                # print_var('src_sent_input_ids', src_sent_input_ids, 'after pegasus padding')
        result = {
            "src_input_ids": src_input_ids, 
            "candidate_ids": candidate_ids,
            "src_sent_ids": src_sent_input_ids,
            "src_input_ids_": src_input_ids,
            "src_attention_mask_": src_attention_mask,
            }
        result["article_tag"] = src_tag,
        if self.is_test:
            result["abstract_tag"] = abs_tag
            result["data"] = data
        return result


def collate_mp_brio(batch, pad_token_id, is_test=False, use_sent=False):
    def pad(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * pad_token_id
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result
    def pad_mask(X, max_len=-1):
        if max_len < 0:
            max_len = max(x.size(0) for x in X)
        result = torch.ones(len(X), max_len, dtype=X[0].dtype) * torch.tensor(0)
        for (i, x) in enumerate(X):
            result[i, :x.size(0)] = x
        return result
    # def pad_sent(X, max_len=-1):
    #     # 对于sent_input_ids进行对齐
    src_input_ids = pad([x["src_input_ids"] for x in batch])
    candidate_ids = [x["candidate_ids"] for x in batch]
    max_len = max([max([len(c) for c in x]) for x in candidate_ids])
    candidate_ids = [pad(x, max_len) for x in candidate_ids]
    # print("candidate_ids : ")
    # print(len(candidate_ids))
    # for i in range(len(candidate_ids)):
    #     print(candidate_ids[i].shape)
    # print("max_len = : ", end="")
    # print(max_len)
    candidate_ids = torch.stack(candidate_ids)
    if use_sent:
        src_sent_ids = [x["src_sent_ids"] for x in batch]
        max_len_sent = max([max([len(c) for c in x]) for x in src_sent_ids])
        src_sent_ids = [pad(x, max_len_sent) for x in src_sent_ids]
        src_input_ids_ = pad([x["src_input_ids_"] for x in batch])
        src_attention_mask_ = pad_mask([x["src_attention_mask_"] for x in batch])
        src_sent_ids = torch.stack(src_sent_ids)
        # print_var('src_sent_ids', src_sent_ids, 'after collate padding')
    else: 
        src_sent_ids = None
        src_input_ids_ = None
        src_attention_mask_ = None
    if is_test:
        data = [x["data"] for x in batch]
        article_tag = [x["article_tag"] for x in batch]
        abstract_tag = [x["abstract_tag"] for x in batch]
    result = {
        "src_input_ids": src_input_ids,
        "candidate_ids": candidate_ids,
        "src_sent_ids": src_sent_ids,
        "src_input_ids_" : src_input_ids_,
        "src_attention_mask_" : src_attention_mask_,
        }
    if is_test:
        result["article_tag"] = article_tag,
        result["data"] = data
        result["abstract_tag"] = abstract_tag,
    return result

def get_sent_txt(article, tags, sent="POSITIVE"):
    res = []
    for sentence, tag in zip(article, tags):
        if tag == sent:
            res.append(sentence)
    sentences = " ".join(res)
    return sentences
    

def cut(obj, sec):
    return [obj[i:i+sec] for i in range(0, len(obj), sec)]

def print_var(var_name, var_value, time):
    print(var_name + ' : ', end="")
    print(var_value, end="")
    print('---------' + time)
    
def get_articles_abscands(data, max_num):
    article = data["article"]
    abstract = data["abstract"]
    candidates = sorted(data["candidates"], key=lambda x:x[1], reverse=True)[0:max_num]
    articles += ' '.join(article)
    articles += '\n'
    abs_cands += abstract[0]
    abs_cands += ' '
    abs_cands += ' '.join([x[0][0] for x in candidates])
    abs_cands += '\n'
    return articles, abs_cands

def write_articles_abscands(articles, abs_cands, article_path, abstract_path, tok_path="/home/hongyucao/PRETRAINED_MODEL/pegasus-x-base", tokenize=False):
    if tokenize:
        tokenizer = PegasusTokenizer.from_pretrained(tok_path)
        with open(article_path+".tokenize", 'w', encoding='utf-8') as f:
            f.write(tokenizer(articles))
        with open(abstract_path+".tokenize", 'w', encoding='utf-8') as f:
            f.write(tokenizer(abs_cands))
    else:
        with open(article_path, 'w', encoding='utf-8') as f:
            f.write(articles)
        with open(abstract_path, 'w', encoding='utf-8') as f:
            f.write(abs_cands)

def get_senti_rate(data):
    #获取一篇文章的情感比重和内容比重，返回一个字典
    senti_list = data["article_tag"]
    senti_set = set(senti_list)
    senti_dict = {}
    for item in senti_set:
        senti_dict.update({item:senti_list.count(item)/len(senti_list)})
    return senti_dict
    


if __name__ == "__main__":
    tok = PegasusTokenizer.from_pretrained("google/pegasus-x-base")
    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=False, use_sent=True)
    train_set = BrioDataset(f"./xsum/diverse_tag/train", "google/pegasus-x-base", max_len=80, max_num=16, total_len=512, is_pegasus=True, sentiment='POSITIVE', use_sent=True)
    dataloader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    for (i, batch) in enumerate(dataloader):
        if i > 5:
            break