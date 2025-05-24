import os
import json
import torch
import logging
from tqdm import tqdm
from torchsummary import summary
from pathlib import Path
from bert_score import score
import logging


device = 'cuda:3'
ptm_path = "/home/hongyucao/PRETRAINED_MODEL/roberta-large"
source_path = Path(r"/home/hongyucao/BRIO-main-Tag/cnndm/diverse_tag")
target_path = Path(r"/home/hongyucao/BRIO-main-Tag/cnndm/diverse_tag")
splits = ['test_0_5', 'test_0_25', 'test_0_75', 'test_1_0']
tag_content = ['article']

def tag(data):
    for content in tag_content:
        if content+'_bert_score' in data.keys():
            break
        # res_tag = []
        # res_logits = []
        res_bert_score = []
        # print(len(data[content]))
        # print(len(data['abstract']))
        abstract = [' '.join(data['abstract'])] #所有abstract整合成一句
        P, R, res_bert_score = score(data[content], abstract*len(data[content]), lang='en', verbose=True, device=device)
        data[content+'_bert_score'] = [r.numpy().tolist() for r in res_bert_score]
        # 估计分数低于0.8的就不能要
    return data


def tag_all():
    for split in splits:
        logging.info('processing '+ split + 'set : ')
        total = len(os.listdir(source_path.joinpath(split)))
        for file in tqdm(source_path.joinpath(split).iterdir(), total=total):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data = tag(data)
            with open(target_path.joinpath(split).joinpath(file.name), 'w', encoding='utf-8') as f:
                json.dump(data, f)

def tag_0():
    data_name = "cnndm"
    src_path0 = "/home/hongyucao/BRIO-main-Tag/"+data_name+"/diverse_tag/test_0_5/1.json"
    tgt_path0 = "/home/hongyucao/BRIO-main-Tag/"+data_name+"/diverse_tag/test_0_5/1.json"
    with open(src_path0, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = tag(data)
    with open(tgt_path0, 'w', encoding='utf-8') as f:
        json.dump(data, f)
                  
if __name__ == "__main__":
    # tag_0()
    tag_all()