import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import BartTokenizer, PegasusTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp_brio, BrioDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from model import RankingLoss, BRIO
import logging
from label_smoothing_loss import label_smoothing_loss
from nltk import sent_tokenize, word_tokenize
from config import cnndm_setting, xsum_setting, base_setting
from tqdm import tqdm
from data_utils import cut
import time
from evaluate_sent import evaluation_sent

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)

def evaluation(args):
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
    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=True)
    test_set = BrioDataset(f"./{args.dataset}/{args.datatype}/test", args.model_type, is_test=True, max_len=512,
     is_sorted=False, max_num=args.max_num, is_untok=True, total_len=args.total_len, is_pegasus=args.is_pegasus)
    batch_size = 4
    dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model = BRIO(model_path, tok.pad_token_id, args.is_pegasus)
    if args.cuda:
        model = model.cuda()

    model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{args.gpuid[0]}'))
    device = f'cuda:{args.gpuid[0]}'
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
        # evaluate the model as a scorer
        mkdir("./result/%s/reference_ranking"%model_name)
        mkdir("./result/%s/candidate_ranking"%model_name)
        rouge1, rouge2, rougeLsum = 0, 0, 0
        cnt = 0
        model.scoring_mode()
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(test_set) // batch_size):
                if args.cuda:
                    to_cuda(batch, args.gpuid[0])
                samples = batch["data"]
                output = model(batch["src_input_ids"], batch["candidate_ids"], args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
                similarity = output['score']
                similarity = similarity.cpu().numpy()
                max_ids = similarity.argmax(1)
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
        print("ranking rouge1: %.6f, rouge2: %.6f, rougeL: %.6f"%(rouge1, rouge2, rougeLsum))

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
                        summaries = model.generate(
                            input_ids=dct["input_ids"].to(device),
                            attention_mask=dct["attention_mask"].to(device),
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
                        input_ids=dct["input_ids"].to(device),
                        attention_mask=dct["attention_mask"].to(device),
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
        
        with open(os.path.join(root_dir, "test.out")) as fout, open(f'./{args.dataset}/{args.datatype}/test.target') as target:
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


def test(dataloader, gen_dataloader, model, args, tok, gpuid, do_sample=False):
    model.eval()
    if args.cuda:
        device = f"cuda:{gpuid}"
    else:
        device = "cpu"
    if len(args.gpuid) > 1:
        _model = model.module
    else:
        _model = model
    cnt = 0
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    mle_loss = 0
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    _model.scoring_mode()
    with torch.no_grad():
        # scoring
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, device)
            samples = batch["data"]
            output = model(batch["src_input_ids"], batch["candidate_ids"], args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity * args.scale
            gold_similarity = gold_similarity * args.scale
            similarity = similarity.cpu().numpy()
            probs = output["probs"]  # [bz, seq_len, word_num]
            probs = output["probs"][:, :-1]  # truncate last token
            gold = batch["candidate_ids"][:, 0, 1:]  # shift right
            mle_loss += mle_fn(probs.transpose(1, 2), gold)
            if i % 1000 == 0:
                print(f"test similarity: {similarity[0]}")
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):
                cnt += 1
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    mle_loss = mle_loss / cnt

    if len(args.gpuid) > 1:
        rouge1 = torch.FloatTensor([rouge1]).to(device)
        dist.all_reduce(rouge1, op=dist.reduce_op.SUM)
        rouge1 = rouge1.item() / len(args.gpuid)
        rouge2 = torch.FloatTensor([rouge2]).to(device)
        dist.all_reduce(rouge2, op=dist.reduce_op.SUM)
        rouge2 = rouge2.item() / len(args.gpuid)
        rougeLsum = torch.FloatTensor([rougeLsum]).to(device)
        dist.all_reduce(rougeLsum, op=dist.reduce_op.SUM)
        rougeLsum = rougeLsum.item() / len(args.gpuid)
        dist.all_reduce(mle_loss, op=dist.reduce_op.SUM)
        mle_loss = mle_loss.item() / len(args.gpuid)
    
    cnt = 0
    sample_rouge1, sample_rouge2, sample_rougeLsum = 0, 0, 0
    if do_sample:
        # generation
        _model.generation_mode()
        def process(x):
            return sent_tokenize(" ".join(word_tokenize(x.strip())))
        with torch.no_grad():
            for (i, batch) in enumerate(gen_dataloader):
                if args.cuda:
                    to_cuda(batch, device)
                samples = batch["data"]
                slines = [" ".join(x["article_untok"]) for x in samples]
                dct = tok.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                summaries = _model.generate(
                    input_ids=dct["input_ids"].to(device),
                    attention_mask=dct["attention_mask"].to(device),
                    max_length=args.gen_max_len + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=args.gen_min_len + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=True,
                )
                dec = [tok.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                for (hypothesis, x) in zip(dec, samples):
                    hypothesis = hypothesis.replace("\n", " ")
                    ref = " ".join(x["abstract_untok"])
                    x = process(ref)
                    y = process(hypothesis)
                    score = rouge_scorer.score("\n".join(x), "\n".join(y))
                    sample_rouge1 += score["rouge1"].fmeasure
                    sample_rouge2 += score["rouge2"].fmeasure
                    sample_rougeLsum += score["rougeLsum"].fmeasure
                    cnt += 1
        _model.scoring_mode()
        sample_rouge1 = sample_rouge1 / cnt
        sample_rouge2 = sample_rouge2 / cnt
        sample_rougeLsum = sample_rougeLsum / cnt
        if len(args.gpuid) > 1:
            sample_rouge1 = torch.FloatTensor([sample_rouge1]).to(device)
            dist.all_reduce(sample_rouge1, op=dist.reduce_op.SUM)
            sample_rouge1 = sample_rouge1.item() / len(args.gpuid)
            sample_rouge2 = torch.FloatTensor([sample_rouge2]).to(device)
            dist.all_reduce(sample_rouge2, op=dist.reduce_op.SUM)
            sample_rouge2 = sample_rouge2.item() / len(args.gpuid)
            sample_rougeLsum = torch.FloatTensor([sample_rougeLsum]).to(device)
            dist.all_reduce(sample_rougeLsum, op=dist.reduce_op.SUM)
            sample_rougeLsum = sample_rougeLsum.item() / len(args.gpuid)
    model.train()
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeLsum": rougeLsum,
        "sample_rouge1": sample_rouge1,
        "sample_rouge2": sample_rouge2,
        "sample_rougeLsum": sample_rougeLsum,
        "mle_loss": mle_loss
        } 


def run(rank, args):
    if args.config == "cnndm":
        cnndm_setting(args)
    elif args.config == "xsum":
        xsum_setting(args)
    else:
        base_setting(args)
    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        id = len(os.listdir("./cache"))
        recorder = Recorder(id, args.log)
    # build dataloader
    if args.is_pegasus:
        tok = PegasusTokenizer.from_pretrained(args.model_type)
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=False, use_sent=args.use_sent)
    collate_fn_val = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=True, use_sent=args.use_sent)
    datatype = args.datatype if not args.use_sent else args.datatype + '_tag'
    train_set = BrioDataset(f"./{args.dataset}/{datatype}/train", args.model_type, max_len=args.max_len, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus, sentiment=args.sent_type, use_sent=args.use_sent, use_cls=args.use_cls)
    val_set = BrioDataset(f"./{args.dataset}/{datatype}/val", args.model_type, is_test=True, max_len=512, is_sorted=False, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus, sentiment=args.sent_type, use_sent=args.use_sent, use_cls=args.use_cls)
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	 train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
    	 val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
        val_gen_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
        val_gen_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    model = BRIO(model_path, tok.pad_token_id, is_pegasus=args.is_pegasus)
    if len(args.model_pt) > 0:
        model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{gpuid}'))
    if args.cuda:
        if is_mp:
            # Using DDP
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=False)
        else:
            model = model.cuda()
    model.train()
    # set the model to scoring mode
    if is_mp:
        model.module.scoring_mode()
    else:
        model.scoring_mode()
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    s_optimizer = optim.Adam(model.parameters())
    if is_master:
        recorder.write_config(args, [model], __file__)
    minimum_ranking_loss = 100
    minimum_mle_loss = 1e5
    all_step_cnt = 0
    if is_mp:
        if is_master:
            id = torch.FloatTensor([id]).to(gpuid)
        else:
            id = torch.zeros(1).to(gpuid)
        dist.all_reduce(id, op=dist.reduce_op.SUM)
        id = int(id.item())
    # define evaluation function
    if args.dataset == "xsum":
        def eval_fn(rouge1, rouge2, rougeLsum):
            return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)
    else:
        def eval_fn(rouge1, rouge2, rougeLsum):
            return 1 - (rouge1 * rouge2 + rougeLsum) / 3
    # start training
    T0 = time.time()
    for epoch in range(args.epoch):
        s_optimizer.zero_grad()
        avg_ranking_loss = 0
        avg_mle_loss = 0
        avg_sent_loss = 0 # 用于衡量情感倾向性
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1
            # forward pass
            src_sent_ids = batch["src_sent_ids"]
            output = model(batch["src_input_ids"], batch["candidate_ids"], args.normalize, args.score_mode, args.length_penalty, adding=args.adding)
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity * args.scale
            gold_similarity = gold_similarity * args.scale
            ranking_loss = RankingLoss(similarity, gold_similarity, args.margin, args.gold_margin, args.gold_weight)
            probs = output["probs"]  # [bz, seq_len, word_num]
            probs = output["probs"][:, :-1]  # truncate last token
            gold = batch["candidate_ids"][:, 0, 1:]  # shift right
            block_len = batch["candidate_ids"].shape[2]
            src_sent_ids_split = torch.split(src_sent_ids, block_len, dim=2)
            gold_sent_list = [x[:, 0, 1:] for x in src_sent_ids_split]
            mle_loss = mle_fn(probs.transpose(1, 2), gold)
            mle_sent_loss = torch.tensor(0.0)
            for i, gold_sent in enumerate(gold_sent_list):
                if i > args.sent_num:
                    break
                if gold_sent.shape[1]+1 != block_len:
                    # print("abort a block")
                    continue
                mle_sent_loss += mle_fn(probs.transpose(1, 2), gold_sent.to(probs.device)).to(mle_sent_loss.device)
            mle_sent_loss = mle_sent_loss/(len(gold_sent_list))
            avg_sent_loss += mle_sent_loss.item()
            loss = args.rank_weight * ranking_loss + args.mle_weight * mle_loss + args.sent_weight * mle_sent_loss
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            avg_mle_loss += mle_loss.item() / args.accumulate_step
            avg_ranking_loss += ranking_loss.item() / args.accumulate_step
            loss.backward()
            if step_cnt == args.accumulate_step:
                # updating
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                # adjust learning rate
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                s_optimizer.step()
                s_optimizer.zero_grad()
            if epoch_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                T1 = time.time()
                # report stats
                print("id: %d"%id)
                print(f"similarity: {similarity[:, :10]}")
                if not args.no_gold:
                    print(f"gold similarity: {gold_similarity}")
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f, avg ranking loss: %.6f, avg mle loss: %.6f, avg sent loss %.6f"
                %(epoch+1, epoch_step, avg_loss / args.report_freq, avg_ranking_loss / args.report_freq, avg_mle_loss / args.report_freq, avg_sent_loss / args.report_freq))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.plot("mle_loss", {"loss": avg_mle_loss / args.report_freq}, all_step_cnt)
                recorder.plot("ranking_loss", {"loss": avg_ranking_loss / args.report_freq}, all_step_cnt)
                recorder.plot("sent_loss", {"loss": avg_sent_loss / args.report_freq}, all_step_cnt)
                recorder.print(f"time: {T1 - T0}")
                recorder.print()
                avg_mle_loss, avg_ranking_loss, avg_loss, avg_sent_loss = 0, 0, 0, 0
            del similarity, gold_similarity, loss, mle_loss, ranking_loss, output, probs

            if all_step_cnt % args.eval_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                # evaluate the model as a scorer
                result = test(val_dataloader, val_gen_dataloader, model, args, tok, gpuid, args.do_sample)
                loss = eval_fn(result["rouge1"], result["rouge2"], result["rougeLsum"])
                if loss < minimum_ranking_loss and is_master:
                    minimum_ranking_loss = loss
                    if is_mp:
                        recorder.save(model.module, "model_ranking.bin")
                    else:
                        recorder.save(model, "model_ranking.bin")
                    recorder.print("best ranking loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                if is_master:
                    recorder.print("val ranking loss: %.6f"%(loss))
                    recorder.print("val ranking rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                    %(result["rouge1"], result["rouge2"], result["rougeLsum"]))
                # evaluate the model as a generator
                if args.do_sample:
                    mle_loss = eval_fn(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"])
                else:
                    mle_loss = result["mle_loss"]
                if mle_loss < minimum_mle_loss and is_master:
                    minimum_mle_loss = mle_loss
                    if is_mp:
                        recorder.save(model.module, "model_generation.bin")
                    else:
                        recorder.save(model, "model_generation.bin")
                    recorder.print("best generation loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                if is_master:
                    recorder.print("val generation loss: %.6f"%(mle_loss))
                    if args.do_sample:
                        recorder.print("val generation rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                        %(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"]))
                # save current model
                if is_master:
                    if is_mp:
                        recorder.save(model.module, "model_cur.bin")
                    else:
                        recorder.save(model, "model_cur.bin")
                    recorder.save(s_optimizer, "optimizer.bin")


def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0, help="gpu ids")
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate model")
    parser.add_argument("-r", "--do_reranking", action="store_true", help="do reranking evaluation")
    parser.add_argument("-g", "--do_generation", action="store_true", help="do generation evaluation")
    parser.add_argument("-l", "--log", action="store_true", help="logging")
    parser.add_argument("-p", "--port", type=int, default=12355, help="port")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    parser.add_argument("--config", default="", type=str, help="config path")
    parser.add_argument("--sent-type", default="POSITIVE", type=str, help="sentiment type wanted")
    parser.add_argument("--sent-weight", type=float, default=0.01) 
    parser.add_argument("--use-sent", default=False, type=bool, help="是否使用情感倾向")
    parser.add_argument("--use-cls", default=False, type=bool, help="是否使用在cls添加等位提示的方式修改情感")
    parser.add_argument("--sent-num", type=int, default=5, help="使用")
    parser.add_argument("--test-part", type=str, default="test", help="使用测试的哪一部分")
    args = parser.parse_args()
    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                if args.use_sent:
                    evaluation_sent(args)
                else:
                    evaluation(args)
        elif len(args.gpuid) == 1:
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)
