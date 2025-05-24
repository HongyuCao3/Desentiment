import os
import argparse
from pathlib import Path
from process_utils import copyfile
from tqdm import tqdm
from logging import Logger
import json

root_path = Path("/home/hongyucao/BRIO-main-Tag")

def rename(args):
    src_path = root_path / Path(args.dataset) / Path(args.datatype) / Path(args.src_path)
    tgt_path = root_path / Path(args.dataset) / Path(args.datatype) / Path(args.tgt_path)
    cnt = 0
    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)
    for file in tqdm(os.listdir(src_path)):
        with open(src_path/file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        with open(tgt_path/(str(cnt)+".json"), 'w', encoding="utf-8") as f:
            json.dump(data, f)
        cnt += 1
    print(cnt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="xsum")
    parser.add_argument("--datatype", type=str, default="diverse_tag")
    parser.add_argument("--src-path", type=str, default="test_0_5")
    parser.add_argument("--tgt-path", type=str, default="test_0_5_rename")
    args = parser.parse_args()
    rename(args)
    