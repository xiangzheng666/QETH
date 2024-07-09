# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
import logging
import sys,json
import numpy as np

def read_answers(filename):
    answers={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            answers[js['idx']]=js['idx']
    return answers

def read_predictions(filename):
    predictions={}
    with open(filename) as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            predictions[js['idx']]=js['answers']
    return predictions

def calculate_scores(answers,predictions):
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
                scores.append(1/(rank+1))
                flag=True
                break
        if flag is False:
            scores.append(0)
    result={}
    result['MRR']=round(np.mean(scores),4)
    result['top10']=round(sum(top10)/len(answers),4)
    result['top5']=round(sum(top5)/len(answers),4)
    result['top1']=round(sum(top1)/len(answers),4)
    return result

