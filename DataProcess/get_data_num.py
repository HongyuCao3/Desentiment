from pathlib import Path
import os
data_split = ["test_1", "test_2", "test_3", "test_4"]

def get_data_num(path):
    num_list = []
    for split in data_split:
        num_list.append(len(os.listdir(path / split)))
    return num_list
                        
if __name__ == "__main__":
    root_path = Path("/home/hongyucao/BRIO-main-Tag")
    xsum_path = Path("xsum/diverse_tag")
    cnndm_path = Path("cnndm/diverse_tag")
    print(get_data_num(root_path / xsum_path))
    print(get_data_num(root_path / cnndm_path))
    