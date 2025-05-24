import torch
import os
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import BartTokenizer, PegasusTokenizer, BartForSequenceClassification
from data_utils import to_cuda, collate_mp_brio, BrioDataset
from torch.utils.data import DataLoader
from functools import partial
from model import RankingLoss, BRIO
from nltk import sent_tokenize, word_tokenize
from config import cnndm_setting, xsum_setting, base_setting
from tqdm import tqdm
from collections import Counter
import logging
level = logging.INFO
logging.basicConfig(level=level)
eval_logger = logging.getLogger("eval-logger")
eval_logger.setLevel(level)

def evaluation_sent(args):
    ptm_path = "valhalla/bart-large-sst2"
    eval_logger.info("load data")
    # load data
    if args.config == "cnndm":
        cnndm_setting(args)
    elif args.config == "xsum":
        xsum_setting(args)
    else:
        base_setting(args)
    if args.is_pegasus:
        tok = PegasusTokenizer.from_pretrained(args.model_type)
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=True, use_sent=args.use_sent)
    test_set = BrioDataset(f"./{args.dataset}/{args.datatype}_tag/{args.test_part}", args.model_type, is_test=True, max_len=512,
     is_sorted=False, max_num=args.max_num, is_untok=True, total_len=args.total_len, is_pegasus=args.is_pegasus,
     use_sent=args.use_sent)
    batch_size = 4
    dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # build models
    eval_logger.info("build model")
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model = BRIO(model_path, tok.pad_token_id, args.is_pegasus)
    classifier = BartForSequenceClassification.from_pretrained(ptm_path)
    tok_classifer = BartTokenizer.from_pretrained(ptm_path)
    if args.cuda:
        model = model.cuda()

    model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{args.gpuid[0]}'))
    device_model = f'cuda:{args.gpuid[0]}'
    device_classifier = f'cuda:{args.gpuid[1]}'
    classifier = classifier.to(device_classifier)
    sent_score = {
        '1': 0.0,
        '2': 0.0,
        '3': 0.0,
        '4': 0.0,
    }
    # print(classifier.config) # 有label2id 和 id2label
    # print(classifier.config.id2label)
    # "id2label": {
    #     "0": "NEGATIVE",
    #     "1": "POSITIVE"
    # }
    model.eval()

    model_name = args.model_pt.split("/")[0]

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    print(model_name)
    root_dir = "./result/%s"%model_name
    mkdir(root_dir)
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    if args.do_reranking:
        eval_logger.info("begin evaluate")
        # evaluate the model as a scorer
        mkdir("./result/%s/reference_ranking"%model_name)
        mkdir("./result/%s/candidate_ranking"%model_name)
        rouge1, rouge2, rougeLsum = 0, 0, 0
        cnt = 0
        model.scoring_mode()
        # model.generation_mode()# 这两个在bart模型里面不兼容，在pegasus中没问题
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(test_set) // batch_size):
                if args.cuda:
                    to_cuda(batch, args.gpuid[0])
                samples = batch["data"]
                article_tags = batch["article_tag"][0]
                abstract_tags = batch["abstract_tag"][0]
                # 这里的abstract_tags中的每个元素都有多个句子的tag
                batch_class = [get_item_class(article_tags[i][0], abstract_tags[i][0], args.sent_type) for i in range(len(abstract_tags))]
                model.scoring_mode()
                output = model(batch["src_input_ids"], batch["candidate_ids"], args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
                model.generation_mode()# TODO:考虑如何修正generate生成多条的问题
                summaries = model.generate(
                            input_ids=batch["src_input_ids_"].to(device_model),
                            attention_mask=batch["src_attention_mask_"].to(device_model),
                            max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                            min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                            no_repeat_ngram_size=3,
                            num_beams=args.num_beams,
                            length_penalty=args.length_penalty,
                            early_stopping=True,
                        )
                dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                dct_gen = tok_classifer.batch_encode_plus(dec, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                dct_gen = dct_gen.to(classifier.device)
                gen_logits = classifier(**dct_gen).logits
                tgt_logits = torch.tensor([[1, 0] for _ in range(batch_size)]) if args.sent_type == "NEGETIVE" else torch.tensor([[0, 1] for _ in range(batch_size)])
                try:
                    batch_sent_score = torch.cosine_similarity(gen_logits, tgt_logits.to(gen_logits.device), dim=1)
                except:
                    print(gen_logits.shape)
                    print(tgt_logits.shape)
                for i_bc, bc in enumerate(batch_class):
                    sent_score[str(bc)] += batch_sent_score[i_bc].item()
                similarity = output['score']
                similarity = similarity.cpu().numpy()
                max_ids = similarity.argmax(1)
                print(sent_score)
                for j in range(similarity.shape[0]):
                    sample = samples[j]
                    sents = sample["candidates"][max_ids[j]][0]
                    # print(" ".join(sents), file=f_out)
                    score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                    rouge1 += score["rouge1"].fmeasure
                    rouge2 += score["rouge2"].fmeasure
                    rougeLsum += score["rougeLsum"].fmeasure
                    with open("./result/%s/candidate_ranking/%d.dec"%(model_name, cnt), "w") as f:
                        for s in sents:
                            print(s, file=f)
                    with open("./result/%s/reference_ranking/%d.ref"%(model_name, cnt), "w") as f:
                        for s in sample["abstract"]:
                            print(s, file=f)
                    cnt += 1
        rouge1 = rouge1 / cnt
        rouge2 = rouge2 / cnt
        rougeLsum = rougeLsum / cnt
        # for key in sent_score.keys():
        #     sent_score[key] = sent_score[key] / cnt
        print("ranking rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))
        print("sent score : ",end="")
        print(sent_score)
        print(sent_score)
        print(sent_score)
        print(sent_score)

    if args.do_generation:
        # evaluate the model as a generator
        rouge1, rouge2, rougeLsum = 0, 0, 0
        tokenizer = tok
        count = 1
        bsz = 8
        model.generation_mode()
        total_num = len(os.listdir(f"./{args.dataset}/{args.datatype}/test"))
        with open(f'./{args.dataset}/{args.datatype}/test.source') as source, open(os.path.join(root_dir, "test.out"), 'w') as fout:
            sline = source.readline().strip()
            slines = [sline]
            for sline in tqdm(source, total=total_num):
                if count % bsz == 0:
                    with torch.no_grad():
                        dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                        dct_sent = tok_classifer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                        summaries = model.generate(
                            input_ids=dct["input_ids"].to(device_model),
                            attention_mask=dct["attention_mask"].to(device_model),
                            max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                            min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                            no_repeat_ngram_size=3,
                            num_beams=args.num_beams,
                            length_penalty=args.length_penalty,
                            early_stopping=True,
                        )
                        source_sent = classifier(dct_sent).logits
                        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                        dct_gen = tok_classifer.batch_encode_plus(dec, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                        gen_sent = classifier(dct_gen).logits
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()
                    slines = []
                sline = sline.strip()
                if len(sline) == 0:
                    sline = " "
                slines.append(sline)
                count += 1
            if slines != []:
                with torch.no_grad():
                    dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                    summaries = model.generate(
                        input_ids=dct["input_ids"].to(device_model),
                        attention_mask=dct["attention_mask"].to(device_model),
                        max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                        min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        early_stopping=True,
                    )
                    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()
        # calculate rouge score
        def process(x):
            return sent_tokenize(" ".join(word_tokenize(x.strip())))
        
        with open(os.path.join(root_dir, "test.out")) as fout, open(f'./{args.dataset}/test.target') as target:
            for (hyp, ref) in zip(fout, target):
                hyp = hyp.strip()
                ref = ref.strip()
                hyp = process(hyp)
                ref = process(ref)
                score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
            rouge1 = rouge1 / total_num
            rouge2 = rouge2 / total_num
            rougeLsum = rougeLsum / total_num
            print("evaluation rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))

def get_sent_from_list(sent_src_list):
    # 最多的元素，如果一样多，则返回中性
    count = Counter(sent_src_list)
    return max(count, key=lambda x: count[x])
 
def get_item_class(sent_src, sent_abs, sent_tgt):
    """根据文本原始情感倾向和标注摘要倾向和目标倾向判断分类

    Args:
        sent_src (Str): 原始文本情感倾向
        sent_abs (Str): 标注摘要情感倾向
        sent_tgt (Str): 目标倾向，字符串
    """
    sent_src_ = get_sent_from_list(sent_src)
    # eval_logger.debug("sent_src : "+sent_src)
    eval_logger.debug("sent_src_ : "+sent_src_)
    eval_logger.debug("sent_abs : "+sent_abs)
    eval_logger.debug("sent_tgt : "+sent_tgt)
    if sent_src_ == sent_abs:
        if sent_abs == sent_tgt:
            return 1
        else:
            return 3
    else:
        if sent_abs == sent_tgt:
            return 2
        else:
            return 4
    
    