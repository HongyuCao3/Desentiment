import os
import json
import torch
import logging
from tqdm import tqdm
from torchsummary import summary
from pathlib import Path
from transformers import BartTokenizer, BartForSequenceClassification
import numpy as np


device = 'cuda:3'
# ptm_path = "ModelTC/bart-base-qnli"
ptm_path = "valhalla/bart-large-sst2"
source_path = Path(r"/home/hongyucao/BRIO-main-Tag/cnndm/diverse")
target_path = Path(r"/home/hongyucao/BRIO-main-Tag/cnndm/diverse_tag")
splits = ['train', 'val', 'test']
tag_content = ['article', 'abstract']
logging.basicConfig(level=logging.INFO)
logging.info('init tokenizer')
tokenzier = BartTokenizer.from_pretrained(ptm_path)
logging.info('init classifier')
classifier = BartForSequenceClassification.from_pretrained(ptm_path).to(device)
print(classifier)
tot_len = 512

def tag(data):
    for content in tag_content:
        res_tag = []
        res_logits = []
        for sentence in data[content]:
            tok_res = tokenzier(sentence, return_tensors='pt', max_length=tot_len).to(device)
            # tok_res = tokenzier.batch_encode_plus([sentence], max_length=tot_len, return_tensors="pt", pad_to_max_length=False, truncation=True)
            with torch.no_grad():
                logits = classifier(**tok_res).logits
            class_id = logits.argmax().item()
            sent_type = classifier.config.id2label[class_id]
            res_tag.append(sent_type)
            res_logits.append(logits.cpu().tolist())
        data[content+'_tag'] = res_tag
        data[content+'_logits'] = res_logits
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
    data_name = "xsum"
    src_path0 = "/home/hongyucao/BRIO-main-Tag/"+data_name+"/diverse/test/0.json"
    tgt_path0 = "/home/hongyucao/BRIO-main-Tag/"+data_name+"/diverse_tag/test/0.json"
    with open(src_path0, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = tag(data)
    with open(tgt_path0, 'w', encoding='utf-8') as f:
        json.dump(data, f)
                  
if __name__ == "__main__":
    tag_0()
    # tag_all()