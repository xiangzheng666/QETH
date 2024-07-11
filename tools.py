import numpy as np
import json
import os

import numpy
import pandas as pd
import logging
import sys,json
import pandas as pd
from tqdm import tqdm

source_vecs = np.load("evaldataset/code_vecs.npy")

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
    aroma = []
    simaple_ndcg10 = []
    for key in tqdm(answers):
        if key not in predictions:
            logging.error("Missing prediction for url {}.".format(key))
            sys.exit()
        flag=False
        dcg10 = 0.00001
        idcg10 = 0.000001
        ndcg10 = 0
        aro = 0
        all_score = np.matmul(source_vecs[key,:],source_vecs.T)
        tmp_score=[all_score[i] for i in predictions[key][:10]]
        for rank,score_rank in enumerate(tmp_score):
            if(rank==0):
                aro = score_rank/100
            dcg10 += score_rank/100/np.log2(rank+2)
        tmp_score = sorted(tmp_score)
        for rank,score_rank in enumerate(tmp_score):
            idcg10 += score_rank/100/np.log2(rank+2)
  
        ndcg10 = dcg10/idcg10
        aroma.append(aro)
        simaple_ndcg10.append(ndcg10)
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
    result['MRR']=str(round(float(numpy.mean(scores)),4))
    result['top10']=str(round(sum(top10)/len(answers),4))
    result['top5']=str(round(sum(top5)/len(answers),4))
    result['top1']=str(round(sum(top1)/len(answers),4))
    result['ndcg10']=str(round(float(numpy.mean(simaple_ndcg10)),4))
    result['aroma']=str(round(float(numpy.mean(aroma)),4))
    return result

def get_result(work_path,flag=True):

    print(work_path+'\n')
    predict_save_file = work_path
    answers = read_answers(predict_save_file,0,10020)
    predictions,score_tmp = read_predictions1(predict_save_file,0,10020)
    scores = calculate_scores(answers, predictions,score_tmp)
    print("ADV-MRR: "+scores['MRR']+"\n")
    print("ADV-ndcg10: "+scores['ndcg10']+"\n")
    print("ADV-aroma: "+scores['aroma']+"\n")
    print("ADV-top10: "+scores['top10']+"\n")
    print("ADV-top5: "+scores['top5']+"\n")
    print("ADV-top1: "+scores['top1']+"\n\n")

    answers = read_answers(predict_save_file, 10020, 10119)
    predictions,score_tmp = read_predictions1(predict_save_file, 10020, 10119)
    scores = calculate_scores(answers, predictions,score_tmp)
    print("CSN-MRR: "+scores['MRR']+"\n")
    print("CSN-ndcg10: "+scores['ndcg10']+"\n")
    print("CSN-aroma: "+scores['aroma']+"\n")
    print("CSN-top10: "+scores['top10']+"\n")
    print("CSN-top5: "+scores['top5']+"\n")
    print("CSN-top1: "+scores['top1']+"\n\n")
    
    answers = read_answers(predict_save_file, 10119, 10642)
    predictions,score_tmp = read_predictions1(predict_save_file, 10119, 10642)
    scores = calculate_scores(answers, predictions,score_tmp)
    print("WEB-MRR: "+scores['MRR']+"\n")
    print("WEB-ndcg10: "+scores['ndcg10']+"\n")
    print("WEB-aroma: "+scores['aroma']+"\n")
    print("WEB-top10: "+scores['top10']+"\n")
    print("WEB-top5: "+scores['top5']+"\n")
    print("WEB-top1: "+scores['top1']+"\n\n")


    answers = read_answers(predict_save_file, 10642, 31396)
    predictions,score_tmp = read_predictions1(predict_save_file, 10642, 31396)
    scores = calculate_scores(answers, predictions,score_tmp)
    print("Xglu-MRR: "+scores['MRR']+"\n")
    print("Xglu-ndcg10: "+scores['ndcg10']+"\n")
    print("Xglu-aroma: "+scores['aroma']+"\n")
    print("Xglu-top10: "+scores['top10']+"\n")
    print("Xglu-top5: "+scores['top5']+"\n")
    print("Xglu-top1: "+scores['top1']+"\n")

print(source_vecs.shape)


# get_result("result/codebertpredictions_qurey_single_1.jsonl")
get_result("out/codebertpredictions_code_desc.jsonl")