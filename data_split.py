'''Split training and dev data from "train.txt"'''
import os
import random
import argparse
import math
SPLIT_RATIO=0.1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, default="data/train.txt")
    parser.add_argument("--tgt_path", type=str, default="data")
    args = parser.parse_args()
    random.seed(42)
    with open(args.src_path,"r") as f:
        lines = f.readlines()

    random.shuffle(lines)
    k = math.ceil(SPLIT_RATIO*len(lines))
    train_split = lines[:-k]
    dev_split = lines[-k:]

    with open(os.path.join(args.tgt_path,"train_split.txt"),"w") as f:
        f.writelines(train_split)
    with open(os.path.join(args.tgt_path,"dev_split.txt"),"w") as f:
        f.writelines(dev_split)