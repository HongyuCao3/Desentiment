import json, os
import argparse
from transformers import PegasusTokenizer
import sys
sys.path.append("..")
import logging
from logging import Logger
from tqdm import tqdm
from collections import Counter
import shutil

def get_senti_rate(data):
    #获取一篇文章的情感比重和内容比重，返回一个字典
    senti_list = data["article_tag"]
    senti_set = set(senti_list)
    senti_dict = {}
    for item in senti_set:
        senti_dict.update({item:senti_list.count(item)/len(senti_list)})
    return senti_dict

def tokenize_test_out(src_path, tgt_path, tok_path, logger):
    """将test.out转换成test.out.tokenized

    Args:
        src_path (str): test.out的路径
        tgt_path (str): test.out.tokenized路径
        tok_path (str): 预训练模型
    """
    logger.info("construct tokenizer")
    tokenizer = PegasusTokenizer.from_pretrained(tok_path)
    logger.info("process test.out")
    res = ""
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            tok_res = tokenizer.tokenize(line)
            res += tok_res
            res += "\n"
    with open(tgt_path, 'w', encoding='utf-8') as f:
        f.write(res)

def get_rate_type(senti_dict, sent_type):
    if sent_type not in senti_dict.keys() or senti_dict[sent_type] <= 0.25:
        return '0.25'
    elif senti_dict[sent_type] <= 0.5:
        return '0.5'
    elif senti_dict[sent_type] <= 0.75:
        return '0.75'
    else:
        return '1'

def get_sent_from_list(sent_src_list):
    # 最多的元素，如果一样多，则返回中性
    count = Counter(sent_src_list)
    return max(count, key=lambda x: count[x])

def get_eval_type(data, sent_type):
    """获取一条训练数据在情感四分类中的具体种类

    Args:
        data (dict): 从json文件中读出的字典
        sent_type (str): 目标情感倾向
    """
    sent_src = get_sent_from_list(data['article_tag'])
    sent_abs = get_sent_from_list(data['abstract_tag'])
    sent_tgt = sent_type
    if sent_src == sent_abs:
        if sent_abs == sent_tgt:
            return '1'
        else:
            return '3'
    else:
        if sent_abs == sent_tgt:
            return '2'
        else:
            return '4'

def get_bert_type(data, sent_type, thres=0.85):
    bert_scores = data['article_bert_score']
    sent_src = data['article_tag']
    gt = 0
    al = 0
    for score, sent in zip(bert_scores, sent_src):
        if score > thres and sent==sent_type:
            gt+=1
        if sent== sent_type:
            al += 1
    if al ==0 :
        return "0.25"
    rate = float(gt)/float(al)
    if rate <=0.25:
        return "0.25"
    elif rate <=0.5:
        return "0.5"
    elif rate <=0.75:
        return "0.75"
    else:
        return "1"

def copyfile(srcfile, dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath))