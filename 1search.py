# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import json

from evaldataset.evaluator import read_answers, read_predictions, calculate_scores
import os

import multiprocessing
from CS.codeSearchModel import Model
from tqdm import tqdm

cpu_cont = multiprocessing.cpu_count()
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 idx,

    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.idx=idx
        
def convert_examples_to_features(js,tokenizer):
    #QETHRD
    if 'code_tokens' in js:
        code=' '.join(js['code_tokens'])
    else:
        code=' '.join(js['function_tokens'])
    code_tokens=tokenizer.tokenize(code)[:256-2]
    code_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids =  tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = 256 - len(code_ids)
    code_ids+=[tokenizer.pad_token_id]*padding_length
    
    nl=' '.join(js['docstring_tokens'])
    nl_tokens=tokenizer.tokenize(nl)[:256-2]
    nl_tokens =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids =  tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = 256 - len(nl_ids)
    nl_ids+=[tokenizer.pad_token_id]*padding_length    
    
    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['idx'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path=None):
        self.examples = []
        data=[]
        with open(file_path) as f:
            for line in f:
                line=line.strip()
                js=json.loads(line)
                data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def test(model, tokenizer,test_file,predict_save_file):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, test_file)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=64)

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0
    code_vecs=[] 
    nl_vecs=[]
    for batch in tqdm(eval_dataloader):
        code_inputs = batch[0].to(device)
        nl_inputs = batch[1].to(device)
        with torch.no_grad():
            code_vec,nl_vec = model(code_inputs,nl_inputs)
            code_vecs.append(code_vec.cpu().numpy())
            nl_vecs.append(nl_vec.cpu().numpy())
        nb_eval_steps += 1
    code_vecs=np.concatenate(code_vecs,0)
    nl_vecs=np.concatenate(nl_vecs,0)
    scores=np.matmul(nl_vecs,code_vecs.T)
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    indexs=[]
 
    for example in eval_dataset.examples:
        indexs.append(example.idx)
    with open(predict_save_file, 'w') as f:
        for index,sort_id in zip(indexs,sort_ids):
            js={}
            js['idx']=index
            js['answers']=[]
            js["score"] = []
            for idx in sort_id[:100]:
                js['answers'].append(int(indexs[idx]))
                js['score'].append(float(scores[index, idx]))
            f.write(json.dumps(js)+'\n')

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def testcodebertdir():
    set_seed()
    config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
    tokenizer = tokenizer_class.from_pretrained("CS/codebert/robatortoken")
    model = model_class.from_pretrained("CS/codebert/model").to(device)
    model = Model(model, tokenizer)
    stat = torch.load("CS/codebert/model/pytorch_model.bin", map_location="cpu")
    del stat['encoder.embeddings.position_ids']
    model.load_state_dict(stat)

    for jsonl in os.listdir("evaldataset/jsonl"):
        name = jsonl.split(".")[0]
        predict_save_file = "result/codebertpredictions_" + name + ".jsonl"
        if os.path.exists(predict_save_file):
            print("codebert --  " + name + ": " + "已经存在")
            continue
        test(model, tokenizer, "evaldataset/jsonl/"+jsonl, predict_save_file)

def testgraphcodebertdir():
    set_seed()
    config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
    tokenizer = tokenizer_class.from_pretrained("CS/graphcodebert/robatortoken")
    model = model_class.from_pretrained("CS/graphcodebert/model").to(device)
    model = Model(model, tokenizer)

    for jsonl in os.listdir("evaldataset/jsonl"):
        name = jsonl.split(".")[0]
        predict_save_file = "result/graphcodebertpredictions_" + name + ".jsonl"
        if os.path.exists(predict_save_file):
            print("graphcodebert --  " + name + ": " + "已经存在")
            continue
        test(model, tokenizer, "evaldataset/jsonl/"+jsonl, predict_save_file)



if __name__ == "__main__":
    testgraphcodebertdir()
    testcodebertdir()


