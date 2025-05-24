import json
from pathlib import Path
from argparse import ArgumentParser

root_dir = Path("/home/hongyucao/BRIO-main-Tag")
model_dir = root_dir / "cache"
arg_target = ["sent_weight", "use_cls", "use_sent", "dataset", "epoch", "sent_type"]

def get_args_map_all(tgt_path):
    """获得所有已经训练模型的名称和参数对应关系
    """
    args_map = {}
    for dir in model_dir.iterdir():
        if dir.is_dir() and (dir/"model_generation.bin").exists():
            args_map.update(
                get_args_map_one(dir)
            )
    args_map_ = sorted(args_map.items(), reverse=False)
    with open(tgt_path, 'w', encoding='utf-8') as f:
        json.dump(args_map_, f)
        

def get_args_map_one(model_path):
    """获得一个模型的名称和参数
    
    Args:
        model_path (str): 模型的保存路径
    """
    config_path = model_path / "config.txt"
    args = get_args_from_config(config_path)
    args_map = {
        str(model_path.name): args
    }
    return args_map
    
def get_args_from_config(config_file):
    """读取config文件并解析namespace

    Args:
        config_file (str): config文件路径
    """
    res = {}
    with open(config_file, 'r', encoding='utf-8') as f:
        config = f.readlines()
        namespace = config[1]
        args = namespace.split('(')[-1][:-1].split(', ')
        # 去掉gpuid的空格
        for arg in args:
            name = arg.split("=")[0]
            if name not in arg_target:
                continue
            value = "".join(arg.split("=")[1:])
            res.update({name : value})
    return res

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tgt-path", type=str, default="./cache/args_map.json")
    args = parser.parse_args()
    # print(get_args_map_one(model_dir/"23-05-28-42"))
    get_args_map_all(args.tgt_path)