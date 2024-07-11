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

    for i in range(min(length,3)):
        if os.path.exists("jsonl/"+name+"_"+str(i+1)+".jsonl"):
            print("jsonl/"+name+"_"+str(i+1)+".jsonl" , " 已经存在")
            continue
        data = []
        index = 0
        for nl, code in zip(nls, codes):
            data.append({
                'idx': index,
                'code_tokens': [i for i in code.split(" ") if i != ""],
                'docstring_tokens': [i for i in nl[i].split(" ") if i != ""],
            })
            index = index + 1

        with open("jsonl/"+name+"_"+str(i+1)+".jsonl", 'w') as f:
            writer = jsonlines.Writer(f)
            for item in data:
                writer.write(item)

def tojsonl2(qureypath,name):
    with open("code.txt", 'r') as f:
        codes = f.read().split("\n")
    with open("qurey.txt", 'r') as f:
        qurey = f.read().split("\n")
    with open(qureypath , 'r') as f:
        nls = f.read().split("\n")

    if os.path.exists("jsonl2/"+name+".jsonl"):
        print("jsonl2/"+name+".jsonl", " 已经存在")
        return
    length = len(nls[0])

    data = []
    index = 0
    for nl, code,q in zip(nls, codes,qurey):
        desc = nl.split("<seq>")
        tmp = {
            'idx': index,
            'nl_length':len(desc),
            'code_tokens': [i for i in code.split(" ") if i != ""],
            'docstring_tokens': [i for i in q.split(" ") if i != ""],
        }
        for n in range(len(desc)):
            tmp['docstring_tokens_'+str(n+1)] =  [i for i in desc[n].split(" ") if i != ""]
        data.append(tmp)
        index = index + 1

    with open("jsonl2/"+name+".jsonl", 'w') as f:
        writer = jsonlines.Writer(f)
        for item in data:
            writer.write(item)

def toqureyjsonl3():
    with open("qurey.txt" , 'r') as f:
        nls = [i.split() for  i in f.read().split("\n")]

    length = len(nls[0])

    for i in range(length):
        data = []
        index = 0
        for nl in nls:
            data.append({
                'url': index,
                'docstring_tokens': nl,
            })
            index = index + 1

        with open("qurey.jsonl", 'w') as f:
            writer = jsonlines.Writer(f)
            for item in data:
                writer.write(item)

if __name__ == '__main__':
    for i in os.listdir("txt"):
        txt = i
        tojsonl("txt/"+txt,txt.split(".")[0])
    for i in os.listdir("txt2"):
        txt = i 
        tojsonl2("txt2/"+txt,txt.split(".")[0])

    toqureyjsonl3()
