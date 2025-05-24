import json, os
import argparse
from transformers import PegasusTokenizer
import sys
sys.path.append("..")
import logging
from logging import Logger
from tqdm import tqdm
from process_utils import get_senti_rate, tokenize_test_out, get_rate_type, get_eval_type, copyfile, get_bert_type

logger = Logger("proces logger", level=logging.INFO)

def process(src_path, article_path, abstract_path, tok_path, max_num=2, tokenize=False):
    file_list = os.listdir(src_path)
    articles = ""
    abs_cands = ""
    # 需要对文本进行分词以得到.tokenize的文件
    logger.info("begin reading data")
    for file in tqdm(file_list):
        with open(src_path+file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if tokenize:
                tokenizer = PegasusTokenizer.from_pretrained(tok_path)
                article = [' '.join(tokenizer.tokenize(x)) for x in data["article"]]
                print(tokenizer.tokenize(data["abstract"][0]))
                abstract = ' '.join(tokenizer.tokenize(data["abstract"][0]))
                # print(sorted(data["candidates"], key=lambda x:x[1], reverse=True)[0:max_num])
                candidates =  [[[' '.join(tokenizer.tokenize(x[0][0]))], x[1]] for x in sorted(data["candidates"], key=lambda x:x[1], reverse=True)[0:max_num]]
            else:
                article = data["article"]
                abstract = data["abstract"]
                candidates = sorted(data["candidates"], key=lambda x:x[1], reverse=True)[0:max_num]
            articles += ' '.join(article)
            articles += '\n'
            abs_cands += abstract[0]
            abs_cands += ' '
            abs_cands += ' '.join([x[0][0] for x in candidates])
            abs_cands += '\n'
        if tokenize:
            with open(article_path, 'w', encoding='utf-8') as f:
                f.write(articles+'.tokenized')
            with open(abstract_path, 'w', encoding='utf-8') as f:
                f.write(abs_cands+'.tokenized')
        else:
            with open(article_path, 'w', encoding='utf-8') as f:
                f.write(articles)
            with open(abstract_path, 'w', encoding='utf-8') as f:
                f.write(abs_cands)

class XSumProcess():
    def __init__(self, args) -> None:
        logger.info("XSumProcess Init...")
        self.src_path = args.src_path
        self.get_test_file = args.get_test_file
        self.get_senti_rate = args.get_senti_rate
        self.article_path = args.article_path
        self.abstract_path = args.abstract_path
        self.tok_path = args.tok_path
        self.tokenzie = args.tokenize
        self.sent_type = args.sent_type
        self.rate_path = args.rate_path
        self.get_type_list = args.get_type_list
        self.type_path = args.type_path
        self.split_test_file = args.split_test_file
        self.data_num = args.data_num
        self.by_senti_rate = args.by_senti_rate
        self.by_bert_score = args.by_bert_score
        
    def process_framework(self):
        file_list = os.listdir(self.src_path)
        rate_cat = {
            '0.25' : [],
            '0.5' : [],
            '0.75' : [],
            '1' : []
        }
        type_list_senti = {
            '1': [],
            '2': [],
            '3': [],
            '4': []
        }
        type_list_bert = {
            '0.25' : [],
            '0.5' : [],
            '0.75' : [],
            '1' : []
        }
        score_cat = {
            '<=0.8': [],
            '>0.8': []
        }
        logger.info("begin process : ")
        cnt = 0
        if self.split_test_file:
            assert os.path.exists(self.type_path) == True
            with open(self.type_path, 'r', encoding='utf-8') as f:
                type_dict = json.load(f)
        for file in tqdm(file_list):
            if self.split_test_file:
                if self.by_senti_rate:
                    keys = ["0.25", "0.5", "0.75", "1"]
                    key2name = {
                        "0.25": "0_25",
                        "0.5": "0_5",
                        "0.75": "0_75",
                        "1": "1_0"
                    }
                elif self.by_bert_score:
                    keys = ["0.25", "0.5", "0.75", "1"]
                    key2name={
                        "0.25": "b_0_25",
                        "0.5": "b_0_5",
                        "0.75": "b_0_75",
                        "1": "b_1_0"
                    }
                else :
                    keys = ["1", "2", "3", "4"]
                for key in keys:
                    if file in type_dict[key]:
                        if self.by_senti_rate :
                            copyfile(self.src_path+file, self.src_path[:-1]+"_"+key2name[key]+"/")
                        elif self.by_bert_score:
                            copyfile(self.src_path+file, self.src_path[:-1]+"/"+key2name[key]+"/")
                        else:
                            copyfile(self.src_path+file, self.src_path[:-1]+"_"+key+"/")
            else:
                with open(self.src_path+file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if self.get_test_file:
                        pass
                    elif self.get_senti_rate:
                        senti_dict = get_senti_rate(data)
                        # 因为测试读取不是batch形式就是句子形式，所以将比例分类之后将数据索引写到一个专门的文件中，便于后续测试
                        rate_cat[get_rate_type(senti_dict, args.sent_type)].append(file)
                    elif self.get_type_list:
                        if self.by_senti_rate:
                            eval_type = get_eval_type(data, args.sent_type)
                            type_list_senti[eval_type].append(file)
                        elif self.by_bert_score:
                            bert_type = get_bert_type(data, args.sent_type)
                            type_list_bert[bert_type].append(file)
            cnt += 1
            if self.data_num != -1 and cnt >= self.data_num :
                break
        logger.info("write in file ...")   
        if self.get_senti_rate:
            with open(self.rate_path, 'w', encoding='utf-8') as f:
                json.dump(rate_cat, f)
        elif self.get_type_list:
            if self.by_senti_rate:
                with open(self.type_path, 'w', encoding='utf-8') as f:
                    json.dump(type_list_senti, f)
            elif self.by_bert_score:
                with open(self.type_path, 'w', encoding='utf-8') as f:
                    json.dump(type_list_bert, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=str)
    parser.add_argument("--tgt-path", type=str, default="")
    parser.add_argument("--article-path", type=str, default="")
    parser.add_argument("--abstract-path", type=str, default="")
    parser.add_argument("--tokenize", action="store_true")
    parser.add_argument("--tok-path", type=str, default="/home/hongyucao/PRETRAINED_MODEL/pegasus-x-base")
    parser.add_argument("--get-test-file", type=bool, default=False)
    parser.add_argument("--get-senti-rate", type=bool, default=False)
    parser.add_argument("--sent-type", type=str, default="POSITIVE")
    parser.add_argument("--rate-path", type=str, default="")
    parser.add_argument("--get-type-list", type=bool, default=False)
    parser.add_argument("--type-path", type=str, default="")
    parser.add_argument("--split-test-file", type=bool, default=False)
    parser.add_argument("--by-senti-rate", type=bool, default=False, help="按照情感比重划分测试集")
    parser.add_argument("--by-bert-score", type=bool, default=False, help="按照语义相似度划分数据集")
    parser.add_argument("--data-num", type=int, default=-1, help="处理文件的数目")
    args = parser.parse_args()
    # process(args.src_path, args.article_path, args.abstract_path, args.tok_path, tokenize=args.tokenize)
    # tokenize_test_out(args.src_path, args.tgt_path, args.tok_path, logger)
    XP = XSumProcess(args)
    XP.process_framework()