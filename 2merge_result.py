import json
import os

import numpy
import pandas as pd
import logging
import sys,json
import pandas as pd
from tqdm import tqdm
def read_answers(filename,start,end):
    answers={}
    with open(filename) as f:
        for line in f.read().split("\n")[start:end]:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['idx']
    return answers

def read_predictions(filename,start,end):
    predictions={}
    scores={}
    with open(filename) as f:
        for line in f.read().split("\n")[start:end]:
            line=line.strip()
            js=json.loads(line)
            predictions[js['idx']]=js['answers']
            scores[js["idx"]]=js['score']
    return predictions,scores

def read_predictions1(filename,start,end):
    predictions={}
    scores={}
    with open(filename) as f:
        for line in f.read().split("\n")[start:end]:
            line=line.strip()
            js=json.loads(line)
            answers_tmp = []
            scores_tmp = []
            for ans,sco in zip(js['answers'],js['score']):
                if ans >= start and ans<end:
                    answers_tmp.append(ans)
                    scores_tmp.append(sco)
            predictions[js['idx']]=answers_tmp
            scores[js["idx"]]=scores_tmp
    return predictions,scores

def calculate_scores(answers,predictions,score):
    scores=[]
    top10=[]
    top5=[]
    top1=[]
    for key in answers:
        if key not in predictions:
            logging.error("Missing prediction for url {}.".format(key))
            sys.exit()
        flag=False
        for rank,idx in enumerate(predictions[key]):
            if idx==answers[key]:
                if(rank<10):
                    top10.append(1)
                if(rank<5):
                    top5.append(1)
                if(rank<1):
                    top1.append(1)
                sc = score[key][rank]
                tmp = rank
                while(tmp>0 and score[key][tmp]==sc):
                    tmp =tmp-1
                scores.append(1/(tmp+1))
                flag=True
                break
        if flag is False:
            scores.append(0)
    result={}
    result['MRR']=round(float(numpy.mean(scores)),4)
    result['top10']=round(sum(top10)/len(answers),4)
    result['top5']=round(sum(top5)/len(answers),4)
    result['top1']=round(sum(top1)/len(answers),4)
    return result

def get_result_from_path(path="result",flag=True):
    if(flag):
        paths = []
        cs = []
        ex = []
        ADV = []
        CSN = []
        WEB = []
        data4 = [] 
        for jsonl in os.listdir(path):
            name = jsonl.split(".")[0]
            cs.append(name.split("_")[0])
            ex.append("_".join(name.split("_")[1:]))
            predict_save_file = path+"/"+jsonl
            answers = read_answers(predict_save_file,0,10020)
            predictions,score_tmp = read_predictions1(predict_save_file,0,10020)
            scores = calculate_scores(answers, predictions,score_tmp)
            ADV.append(scores["MRR"])
            answers = read_answers(predict_save_file, 10020, 10119)
            predictions,score_tmp = read_predictions1(predict_save_file, 10020, 10119)
            scores = calculate_scores(answers, predictions,score_tmp)
            CSN.append(scores["MRR"])
            answers = read_answers(predict_save_file, 10119, 10642)
            predictions,score_tmp = read_predictions1(predict_save_file, 10119, 10642)
            scores = calculate_scores(answers, predictions,score_tmp)
            WEB.append(scores["MRR"])
            answers = read_answers(predict_save_file, 10642, 31396)
            predictions,score_tmp = read_predictions1(predict_save_file, 10642, 31396)
            scores = calculate_scores(answers, predictions,score_tmp)
            data4.append(scores["MRR"])
            paths.append(path)
            print(name)
            print("    ", "ADV:", ADV[-1], "CSN:", CSN[-1], "WEB:", WEB[-1], "data4:", data4[-1])
        return {
            "cs":cs,
            "ex":ex,
            "ADV":ADV,
            "CSN": CSN,
            "WEB": WEB,
            "data4": data4,
            "paths": paths,
        }
    else:
        paths = []
        cs = []
        ex = []
        mrr = [] 
        for jsonl in os.listdir(path):
            name = jsonl.split(".")[0]
            cs.append(name.split("_")[0])
            ex.append("_".join(name.split("_")[1:]))
            predict_save_file = path+"/"+jsonl
            answers = read_answers(predict_save_file,0,31396)
            predictions,score_tmp = read_predictions(predict_save_file,0,31396)
            scores = calculate_scores(answers, predictions,score_tmp)
            mrr.append(scores["MRR"])
            paths.append(path)
            print(name)
            print("    ", "mrr:", mrr[-1])
        return {
            "cs":cs,
            "ex":ex,
            "mrr": mrr,
            "paths": paths,
        }

def getsortscorelist():
    jsonls = [i.split(".")[0] for i in os.listdir("evaldataset/jsonl")]

    resultdata = get_result_from_path("result",False)
    
    df = pd.DataFrame(resultdata)

    all = []
    mrr = []
    for i in range(len(tqdm(df))):
        MRR = df.loc[i, "mrr"]
        js = {}
        js["MRR"] = MRR
        all.append(js)
        mrr.append(MRR)
        
    df["all"] = all
    df["mrr"] = mrr

    ax = df[df['cs'] == "graphcodebertpredictions"][["mrr", "ex"]].sort_values(by="mrr", ascending=False)
    l = ax["ex"].tolist()

    graphcodebert = []
    for i in l:
        k = jsonls.index(i)
        graphcodebert.append(k)
    print(graphcodebert)

    ax = df[df['cs'] == "codebertpredictions"][["mrr", "ex"]].sort_values(by="mrr", ascending=False)
    l = ax["ex"].tolist()

    codebert = []
    for i in l:
        if i in jsonls:
            k = jsonls.index(i)
            codebert.append(k)
   
    print(codebert)

    return codebert, graphcodebert

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
                answers = query[i]['answers'][:50]
                score = query[i]['score'][:50]
                for p in path[:num]:
                    answers = answers + p[i]['answers'][:50]
                    score = score + p[i]['score'][:50]

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
                answers = query[i]['answers'][:50]
                score = query[i]['score'][:50]
                for p in path[:num]:
                    answers = answers + p[i]['answers'][:50]
                    score = score + p[i]['score'][:50]
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
                answers = query[i]['answers'][:50]
                score = query[i]['score'][:50]
                for p in path:
                    answers = answers + p[i]['answers'][:50]
                    score = score + p[i]['score'][:50]

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
                answers = query[i]['answers'][:50]
                score = query[i]['score'][:50]
                for p in path:
                    answers = answers + p[i]['answers'][:50]
                    score = score + p[i]['score'][:50]

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
                answers = query[i]['answers'][:50]
                score = query[i]['score'][:50]
                querymrr = 0
                if idx in answers:
                    querymrr = 1/(answers.index(idx)+1)
                for p in path[:num]:
                    mrr = 0
                    if idx in p[i]['answers'][:50]:
                        mrr = 1/(p[i]['answers'][:50].index(idx)+1)
                    if(mrr>querymrr):
                        querymrr = mrr
                        answers = p[i]['answers'][:50]
                        score =  p[i]['score'][:50]
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
                answers = query[i]['answers'][:50]
                score = query[i]['score'][:50]
                querymrr = 0
                if idx in answers:
                    querymrr = 1/(answers.index(idx)+1)
                for p in path[:num]:
                    mrr = 0
                    if idx in p[i]['answers'][:50]:
                        mrr = 1/(p[i]['answers'][:50].index(idx)+1)
                    if(mrr>querymrr):
                        querymrr = mrr
                        answers = p[i]['answers'][:50]
                        score =  p[i]['score'][:50]
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
                answers = query[i]['answers'][:50]
                score = query[i]['score'][:50]
                querymrr = 0
                if idx in answers:
                    querymrr = 1/(answers.index(idx)+1)
                for p in path[:num]:
                    mrr = 0
                    if idx in p[i]['answers'][:50]:
                        mrr = 1/(p[i]['answers'][:50].index(idx)+1)
                    if(mrr>querymrr):
                        querymrr = mrr
                        answers = p[i]['answers'][:50]
                        score =  p[i]['score'][:50]
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
                answers = query[i]['answers'][:50]
                score = query[i]['score'][:50]
                querymrr = 0
                if idx in answers:
                    querymrr = 1/(answers.index(idx)+1)
                for p in path:
                    mrr = 0
                    if idx in p[i]['answers'][:50]:
                        mrr = 1/(p[i]['answers'][:50].index(idx)+1)
                    if(mrr>querymrr):
                        querymrr = mrr
                        answers = p[i]['answers'][:50]
                        score =  p[i]['score'][:50]
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
                answers = query[i]['answers'][:50]
                score = query[i]['score'][:50]
                querymrr = 0
                if idx in answers:
                    querymrr = 1/(answers.index(idx)+1)
                for p in path:
                    mrr = 0
                    if idx in p[i]['answers'][:50]:
                        mrr = 1/(p[i]['answers'][:50].index(idx)+1)
                    if(mrr>querymrr):
                        querymrr = mrr
                        answers = p[i]['answers'][:50]
                        score =  p[i]['score'][:50]
                js['idx'] = idx
                js['answers'] = answers
                js["score"] = score
                f.write(json.dumps(js) + '\n')

if __name__ == '__main__':
    # codebert, graphcodebert = getsortscorelist()
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


