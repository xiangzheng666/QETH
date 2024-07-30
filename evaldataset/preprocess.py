import json
import random
import time

import jsonlines
import os,re
import requests
from tqdm import tqdm


def tojsonl(qureypath,name):
    with open("code.txt", 'r') as f:
        codes = f.read().split("\n")
    with open(qureypath , 'r') as f:
        nls = [i.split("<seq>") for  i in f.read().split("\n")]
        
    
    length = len(nls[0])

    for index_num in range(min(length,3)):
        
        if os.path.exists("jsonl/"+name+"_"+str(index_num+1)+".jsonl"):
            print("jsonl/"+name+"_"+str(index_num+1)+".jsonl" , " 已经存在")
            continue
        data = []
        index = 0
        print("jsonl/"+name+"_"+str(index_num+1)+".jsonl" , " 未存在，解析中")
        for nl, code in zip(nls, codes):
            data.append({
                'idx': index,
                'code_tokens': [i for i in code.split(" ") if i != ""],
                'docstring_tokens': [i for i in nl[index_num].split(" ") if i != ""],
            })
            index = index + 1

        with open("jsonl/"+name+"_"+str(index_num+1)+".jsonl", 'w') as f:
            writer = jsonlines.Writer(f)
            for item in data:
                writer.write(item)

if __name__ == '__main__':
    for i in os.listdir("txt"):
        txt = i
        tojsonl("txt/"+txt,txt.split(".")[0])
