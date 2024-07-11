import json
import os

import numpy
import pandas as pd
import logging
import sys,json
import pandas as pd
from tqdm import tqdm

def getjson(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            data.append(js)
    return data

def mergergraphcodebert(num=10, sortlist=None):
    query = getjson("result/graphcodebertpredictions_qurey_single_1.jsonl")
    jsonls = os.listdir("evaldataset/jsonl")
    if sortlist is None:
        sortlist = [1,16,29,7,9,22,5,34,23,10,28,6,25,32,13,3,24,21,30,33,8,0,19,17,20,15,31,2,18,12,11,4,26,27,14]
    for txt in os.listdir("evaldataset/txt"):
        name = txt.split(".")[0]
        print(name)
        path = []
        if os.path.exists("out/graphcodebertpredictions_"+name+".jsonl"):
            print("graphcodebert --  " + name + ": " + "已经存在")
            continue
        for p in sortlist:
            if jsonls[p].startswith(name):
                path.append(getjson("result/graphcodebertpredictions_"+jsonls[p]))
                print("result/graphcodebertpredictions_"+jsonls[p])

        with open("out/graphcodebertpredictions_"+name+".jsonl",'w+') as f:
            for i in tqdm(range(len(query))):
                js = {}
                idx = query[i]['idx']
                answers = query[i]['answers'][:100]
                score = query[i]['score'][:100]
                for p in path[:num]:
                    answers = answers + p[i]['answers'][:100]
                    score = score + p[i]['score'][:100]

                answers = numpy.array(answers)
                score = numpy.array(score)
                tmp = numpy.argsort(score)
                score = score[tmp[::-1]].tolist()
                answers = answers[tmp[::-1]].tolist()
                answers1 = []
                score1 = []
                pre = []
                for i in range(len(answers)):
                    if answers[i] in pre:
                        continue
                    score1.append(score[i])
                    answers1.append(answers[i])
                    pre.append(answers[i])
                js['idx'] = idx
                js['answers'] = answers1
                js["score"] = score1
                f.write(json.dumps(js) + '\n')

def mergercodebert(num=10, sortlist=None):

    query = getjson("result/codebertpredictions_qurey_single_1.jsonl")
    jsonls = os.listdir("evaldataset/jsonl")
    if sortlist is None:
        sortlist = [1,29,16,34,3,7,5,22,9,0,24,28,10,33,19,6,13,23,21,8,25,30,32,11,26,2,31,15,17,27,12,18,4,14,20]
    for txt in os.listdir("evaldataset/txt"):
        name = txt.split(".")[0]
        print(name)
        path = []
        if os.path.exists("out/codebertpredictions_"+name+".jsonl"):
            print("codebert --  " + name + ": " + "已经存在")
            continue
        for p in sortlist:
            if jsonls[p].startswith(name):
                path.append(getjson("result/codebertpredictions_" + jsonls[p]))
                print("result/codebertpredictions_"+jsonls[p])
                
        with open("out/codebertpredictions_"+name+".jsonl",'w+') as f:
            for i in tqdm(range(len(query))):
                js = {}
                idx = query[i]['idx']
                answers = query[i]['answers'][:100]
                score = query[i]['score'][:100]
                for p in path[:num]:
                    answers = answers + p[i]['answers'][:100]
                    score = score + p[i]['score'][:100]
                answers = numpy.array(answers)
                score = numpy.array(score)
                tmp = numpy.argsort(score)
                score = score[tmp[::-1]].tolist()
                answers = answers[tmp[::-1]].tolist()
                answers1 = []
                score1 = []
                pre = []
                for i in range(len(answers)):
                    if answers[i] in pre:
                        continue
                    score1.append(score[i])
                    answers1.append(answers[i])
                    pre.append(answers[i])
                js['idx'] = idx
                js['answers'] = answers1
                js["score"] = score1
                f.write(json.dumps(js) + '\n')

def mergergraphcodebertjsonl2():
    query = getjson("result/graphcodebertpredictions_qurey_single_1.jsonl")
    
    for txt in os.listdir("evaldataset/txt2"):
        name = txt.split(".")[0]
        print(name)
        path = []
        if os.path.exists("out/graphcodebertpredictions_"+name+"_jsonl2.jsonl"):
            print("graphcodebert --  " + name + " _jsonl2 : " + "已经存在")
            continue
        
        path.append(getjson("out/graphcodebertpredictions_"+name+'.jsonl'))
        print("out/graphcodebertpredictions_"+name+'.jsonl')

        with open("out/graphcodebertpredictions_"+name+"_jsonl2.jsonl",'w+') as f:
            for i in tqdm(range(len(query))):
                js = {}
                idx = query[i]['idx']
                answers = query[i]['answers'][:100]
                score = query[i]['score'][:100]
                for p in path:
                    answers = answers + p[i]['answers'][:100]
                    score = score + p[i]['score'][:100]

                answers = numpy.array(answers)
                score = numpy.array(score)
                tmp = numpy.argsort(score)
                score = score[tmp[::-1]].tolist()
                answers = answers[tmp[::-1]].tolist()
                answers1 = []
                score1 = []
                pre = []
                for i in range(len(answers)):
                    if answers[i] in pre:
                        continue
                    score1.append(score[i])
                    answers1.append(answers[i])
                    pre.append(answers[i])
                js['idx'] = idx
                js['answers'] = answers1
                js["score"] = score1
                f.write(json.dumps(js) + '\n')

def mergercodebertjsonl2():
    query = getjson("result/codebertpredictions_qurey_single_1.jsonl")
    
    for txt in os.listdir("evaldataset/txt2"):
        name = txt.split(".")[0]
        print(name)
        path = []
        if os.path.exists("out/codebertpredictions_"+name+"_jsonl2.jsonl"):
            print("codebert --  " + name + " _jsonl2 : " + "已经存在")
            continue
        
        path.append(getjson("out/codebertpredictions_"+name+'.jsonl'))
        print("out/codebertpredictions_"+name+'.jsonl')

        with open("out/codebertpredictions_"+name+"_jsonl2.jsonl",'w+') as f:
            for i in tqdm(range(len(query))):
                js = {}
                idx = query[i]['idx']
                answers = query[i]['answers'][:100]
                score = query[i]['score'][:100]
                for p in path:
                    answers = answers + p[i]['answers'][:100]
                    score = score + p[i]['score'][:100]

                answers = numpy.array(answers)
                score = numpy.array(score)
                tmp = numpy.argsort(score)
                score = score[tmp[::-1]].tolist()
                answers = answers[tmp[::-1]].tolist()
                answers1 = []
                score1 = []
                pre = []
                for i in range(len(answers)):
                    if answers[i] in pre:
                        continue
                    score1.append(score[i])
                    answers1.append(answers[i])
                    pre.append(answers[i])
                js['idx'] = idx
                js['answers'] = answers1
                js["score"] = score1
                f.write(json.dumps(js) + '\n')

def mergergraphcodebert_best(num=10, sortlist=None):
    query = getjson("result/graphcodebertpredictions_qurey_single_1.jsonl")
    jsonls = os.listdir("evaldataset/jsonl")
    if sortlist is None:
        sortlist = [1,16,29,7,9,22,5,34,23,10,28,6,25,32,13,3,24,21,30,33,8,0,19,17,20,15,31,2,18,12,11,4,26,27,14]
    for txt in os.listdir("evaldataset/txt"):
        name = txt.split(".")[0]
        print(name)
        path = []
        if os.path.exists("out2/graphcodebertpredictions_"+name+".jsonl"):
            print("graphcodebert --  " + name + ": " + "已经存在")
            continue
        for p in sortlist:
            if jsonls[p].startswith(name):
                path.append(getjson("result/graphcodebertpredictions_"+jsonls[p]))
                print("result/graphcodebertpredictions_"+jsonls[p])

        with open("out2/graphcodebertpredictions_"+name+".jsonl",'w+') as f:
            for i in tqdm(range(len(query))):
                js = {}
                idx = query[i]['idx']
                answers = query[i]['answers'][:100]
                score = query[i]['score'][:100]
                querymrr = 0
                if idx in answers:
                    querymrr = 1/(answers.index(idx)+1)
                for p in path[:num]:
                    mrr = 0
                    if idx in p[i]['answers'][:100]:
                        mrr = 1/(p[i]['answers'][:100].index(idx)+1)
                    if(mrr>querymrr):
                        querymrr = mrr
                        answers = p[i]['answers'][:100]
                        score =  p[i]['score'][:100]
                js['idx'] = idx
                js['answers'] = answers
                js["score"] = score
                f.write(json.dumps(js) + '\n')

def mergercodebert_best(num=10, sortlist=None):
    query = getjson("result/codebertpredictions_qurey_single_1.jsonl")
    jsonls = os.listdir("evaldataset/jsonl")
    if sortlist is None:
        sortlist = [1,16,29,7,9,22,5,34,23,10,28,6,25,32,13,3,24,21,30,33,8,0,19,17,20,15,31,2,18,12,11,4,26,27,14]
    for txt in os.listdir("evaldataset/txt"):
        name = txt.split(".")[0]
        print(name)
        path = []
        if os.path.exists("out2/codebertpredictions_"+name+".jsonl"):
            print("codebert --  " + name + ": " + "已经存在")
            continue
        for p in sortlist:
            if jsonls[p].startswith(name):
                path.append(getjson("result/codebertpredictions_"+jsonls[p]))
                print("result/codebertpredictions_"+jsonls[p])

        with open("out2/codebertpredictions_"+name+".jsonl",'w+') as f:
            for i in tqdm(range(len(query))):
                js = {}
                idx = query[i]['idx']
                answers = query[i]['answers'][:100]
                score = query[i]['score'][:100]
                querymrr = 0
                if idx in answers:
                    querymrr = 1/(answers.index(idx)+1)
                for p in path[:num]:
                    mrr = 0
                    if idx in p[i]['answers'][:100]:
                        mrr = 1/(p[i]['answers'][:100].index(idx)+1)
                    if(mrr>querymrr):
                        querymrr = mrr
                        answers = p[i]['answers'][:100]
                        score =  p[i]['score'][:100]
                js['idx'] = idx
                js['answers'] = answers
                js["score"] = score
                f.write(json.dumps(js) + '\n')

def mergercodebert_best_n(nums=10, sortlist=None):
    query = getjson("result/codebertpredictions_qurey_single_1.jsonl")
    jsonls = os.listdir("evaldataset/jsonl")
    if sortlist is None:
        sortlist = [1,16,29,7,9,22,5,34,23,10,28,6,25,32,13,3,24,21,30,33,8,0,19,17,20,15,31,2,18,12,11,4,26,27,14]

    txt = "sequre_single.txt"
    name = txt.split(".")[0]
    print(name)
    path = []
    for p in sortlist:
        if jsonls[p].startswith(name):
            path.append(getjson("result/codebertpredictions_"+jsonls[p]))
            print("result/codebertpredictions_"+jsonls[p])
    for num in range(1,nums+1):
        if os.path.exists("out3/codebertpredictions_"+name+"_"+str(num)+".jsonl"):
            print("codebert --  " + name + ": " + "已经存在")
            continue
        with open("out3/codebertpredictions_"+name+"_"+str(num)+".jsonl",'w+') as f:
            for i in tqdm(range(len(query))):
                js = {}
                idx = query[i]['idx']
                answers = query[i]['answers'][:100]
                score = query[i]['score'][:100]
                querymrr = 0
                if idx in answers:
                    querymrr = 1/(answers.index(idx)+1)
                for p in path[:num]:
                    mrr = 0
                    if idx in p[i]['answers'][:100]:
                        mrr = 1/(p[i]['answers'][:100].index(idx)+1)
                    if(mrr>querymrr):
                        querymrr = mrr
                        answers = p[i]['answers'][:100]
                        score =  p[i]['score'][:100]
                js['idx'] = idx
                js['answers'] = answers
                js["score"] = score
                f.write(json.dumps(js) + '\n')

def mergercodebertjsonl2_bset():
    query = getjson("result/codebertpredictions_qurey_single_1.jsonl")
    
    for txt in os.listdir("evaldataset/txt2"):
        name = txt.split(".")[0]
        print(name)
        path = []
        if os.path.exists("out2/codebertpredictions_"+name+"_jsonl2.jsonl"):
            print("codebert --  " + name + " _jsonl2 : " + "已经存在")
            continue
        
        path.append(getjson("out/codebertpredictions_"+name+'.jsonl'))
        print("out/codebertpredictions_"+name+'.jsonl')

        with open("out2/codebertpredictions_"+name+"_jsonl2.jsonl",'w+') as f:
            for i in tqdm(range(len(query))):
                js = {}
                idx = query[i]['idx']
                answers = query[i]['answers'][:100]
                score = query[i]['score'][:100]
                querymrr = 0
                if idx in answers:
                    querymrr = 1/(answers.index(idx)+1)
                for p in path:
                    mrr = 0
                    if idx in p[i]['answers'][:100]:
                        mrr = 1/(p[i]['answers'][:100].index(idx)+1)
                    if(mrr>querymrr):
                        querymrr = mrr
                        answers = p[i]['answers'][:100]
                        score =  p[i]['score'][:100]
                js['idx'] = idx
                js['answers'] = answers
                js["score"] = score
                f.write(json.dumps(js) + '\n')

def mergergraphcodebertjsonl2_bset():
    query = getjson("result/graphcodebertpredictions_qurey_single_1.jsonl")
    
    for txt in os.listdir("evaldataset/txt2"):
        name = txt.split(".")[0]
        print(name)
        path = []
        if os.path.exists("out2/graphcodebertpredictions_"+name+"_jsonl2.jsonl"):
            print("codebert --  " + name + " _jsonl2 : " + "已经存在")
            continue
        
        path.append(getjson("out/graphcodebertpredictions_"+name+'.jsonl'))
        print("out/graphcodebertpredictions_"+name+'.jsonl')

        with open("out2/graphcodebertpredictions_"+name+"_jsonl2.jsonl",'w+') as f:
            for i in tqdm(range(len(query))):
                js = {}
                idx = query[i]['idx']
                answers = query[i]['answers'][:100]
                score = query[i]['score'][:100]
                querymrr = 0
                if idx in answers:
                    querymrr = 1/(answers.index(idx)+1)
                for p in path:
                    mrr = 0
                    if idx in p[i]['answers'][:100]:
                        mrr = 1/(p[i]['answers'][:100].index(idx)+1)
                    if(mrr>querymrr):
                        querymrr = mrr
                        answers = p[i]['answers'][:100]
                        score =  p[i]['score'][:100]
                js['idx'] = idx
                js['answers'] = answers
                js["score"] = score
                f.write(json.dumps(js) + '\n')

if __name__ == '__main__':

    codebert = [i for i in range(len(os.listdir("evaldataset/jsonl")))]
    graphcodebert = [i for i in range(len(os.listdir("evaldataset/jsonl")))]
    
    # 将结果全局排序合并
    mergergraphcodebert(10,graphcodebert)
    mergercodebert(10,codebert)

    # 将merger的结果全局排序合并
    mergercodebertjsonl2()
    mergergraphcodebertjsonl2()
    
    
    #测试最好结果影响
    mergergraphcodebert_best(10,graphcodebert)
    mergercodebert_best(10,codebert)
    mergercodebertjsonl2_bset()
    mergergraphcodebertjsonl2_bset()
    
    #测试n对最好结果影响
    # mergercodebert_best_n(10,codebert)


