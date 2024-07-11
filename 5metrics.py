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

def get_result_from_path(work_path="result",flag=True):

    for jsonl in os.listdir(work_path):
        path = os.path.join("metrics",work_path+"_"+jsonl.split(".")[0]+".txt")
        if os.path.exists(path):
            print(path , " 已经存在")
            continue
        with open(path,'w+') as f:
            name = jsonl.split(".")[0]
            predict_save_file = work_path+"/"+jsonl
            answers = read_answers(predict_save_file,0,10020)
            predictions,score_tmp = read_predictions1(predict_save_file,0,10020)
            scores = calculate_scores(answers, predictions,score_tmp)
            f.write("ADV-MRR: "+scores['MRR']+"\n")
            f.write("ADV-ndcg10: "+scores['ndcg10']+"\n")
            f.write("ADV-aroma: "+scores['aroma']+"\n")
            f.write("ADV-top10: "+scores['top10']+"\n")
            f.write("ADV-top5: "+scores['top5']+"\n")
            f.write("ADV-top1: "+scores['top1']+"\n\n")

            answers = read_answers(predict_save_file, 10020, 10119)
            predictions,score_tmp = read_predictions1(predict_save_file, 10020, 10119)
            scores = calculate_scores(answers, predictions,score_tmp)
            f.write("CSN-MRR: "+scores['MRR']+"\n")
            f.write("CSN-ndcg10: "+scores['ndcg10']+"\n")
            f.write("CSN-aroma: "+scores['aroma']+"\n")
            f.write("CSN-top10: "+scores['top10']+"\n")
            f.write("CSN-top5: "+scores['top5']+"\n")
            f.write("CSN-top1: "+scores['top1']+"\n\n")
            
            answers = read_answers(predict_save_file, 10119, 10642)
            predictions,score_tmp = read_predictions1(predict_save_file, 10119, 10642)
            scores = calculate_scores(answers, predictions,score_tmp)
            f.write("WEB-MRR: "+scores['MRR']+"\n")
            f.write("WEB-ndcg10: "+scores['ndcg10']+"\n")
            f.write("WEB-aroma: "+scores['aroma']+"\n")
            f.write("WEB-top10: "+scores['top10']+"\n")
            f.write("WEB-top5: "+scores['top5']+"\n")
            f.write("WEB-top1: "+scores['top1']+"\n\n")


            answers = read_answers(predict_save_file, 10642, 31396)
            predictions,score_tmp = read_predictions1(predict_save_file, 10642, 31396)
            scores = calculate_scores(answers, predictions,score_tmp)
            f.write("Xglu-MRR: "+scores['MRR']+"\n")
            f.write("Xglu-ndcg10: "+scores['ndcg10']+"\n")
            f.write("Xglu-aroma: "+scores['aroma']+"\n")
            f.write("Xglu-top10: "+scores['top10']+"\n")
            f.write("Xglu-top5: "+scores['top5']+"\n")
            f.write("Xglu-top1: "+scores['top1']+"\n")

            answers = read_answers(predict_save_file, 31396, 53572)
            predictions,score_tmp = read_predictions1(predict_save_file, 31396, 53572)
            scores = calculate_scores(answers, predictions,score_tmp)
            f.write("CSN-TEST-MRR: "+scores['MRR']+"\n")
            f.write("CSN-TEST-ndcg10: "+scores['ndcg10']+"\n")
            f.write("CSN-TEST-aroma: "+scores['aroma']+"\n")
            f.write("CSN-TEST-top10: "+scores['top10']+"\n")
            f.write("CSN-TEST-top5: "+scores['top5']+"\n")
            f.write("CSN-TEST-top1: "+scores['top1']+"\n")

            

    

print(source_vecs.shape)

get_result_from_path("out")
get_result_from_path("out2")
get_result_from_path("result")

