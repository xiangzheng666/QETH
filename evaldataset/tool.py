import json

import nltk

from nltk.corpus import wordnet as wn

def getnn(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    index = [0 for i in range(len(tagged))]
    sentence = [i[0] for i in tagged]
    for i in range(len(tagged)):
        # if "NN" in tagged[i][1]
        tmp = tagged[i]
        if "NN" in tmp[1] or "JJ" in tmp[1]:
            index[i] = 1
    return sentence,index

def getSimilarity(qurey,source):

    qurey,qureyindex = getnn(qurey)
    source, sourceindex = getnn(source)

    for q in range(len(qurey)):
        if(qureyindex[q]==0):
            continue
        for s in range(len(source)):
            if(sourceindex[s]==0):
                continue
            
            qword = wn.synsets(qurey[q])
            sword = wn.synsets(source[s])

            if len(qword)==0 or len(sword)==0:
                continue

            score = qword[0].path_similarity(sword[0])
            if score>0.3:
                qureyindex[q] = 0
                sourceindex[s] = 0
    qureydata = [qurey[i] for i in range(len(qurey)) if qureyindex[i]==1]
    index = 0
    for s in range(len(source)):
        if(sourceindex[s]==1):
            if(index<len(qureydata)):
                source[s] = qureydata[index]
                sourceindex[s] = 0
                index += 1
            else:
                sourceindex[s] = -2
                index += 1
    source.extend(qureydata[index+1:])
    sourceindex.extend([0 for i in range(len(qureydata)-index-1)])
    return [source[i] for i in range(len(source)) if sourceindex[i]==0]


with open('source.txt','r') as f:
    source = f.read().split('\n')
with open('qurey.txt','r') as f:
    qurey = f.read().split('\n')

data=[]

from tqdm import tqdm

with open('desc.jsonl','r') as f:
    for line in tqdm(f.readlines()):
        tmp = []
        tmp2 = []
        js=json.loads(line)
        q = qurey[js["idx"]].lower()
        answers = js['answers']
        scores = js['score']
        for i,score in zip(answers,scores):
            if len(tmp)==3:
                break
            if source[int(i)] not in tmp:
                if float(score)>=90:
                    tmp.append(source[int(i)].lower())
                else:
                    s = source[int(i)].lower()
                    s = getSimilarity(q,s)
                    tmp.append(" ".join(s))
        data.append('<seq>'.join(tmp))

with open('siqe_source.txt','w') as f:
    f.write('\n'.join(data))