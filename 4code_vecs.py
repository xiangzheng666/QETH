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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 nl_tokens,
                 nl_ids,
                 idx
    ):
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.idx = idx
        
def convert_examples_to_features(nl,tokenizer,idx):

    nl_tokens=tokenizer.tokenize(nl)[:256-2]
    nl_tokens0 =[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids0 =  tokenizer.convert_tokens_to_ids(nl_tokens0)
    padding_length = 256 - len(nl_ids0)
    nl_ids0+=[tokenizer.pad_token_id]*padding_length  

    return InputFeatures(nl_tokens0,nl_ids0,idx)

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path=None):
        self.examples = []
        data=[]
        with open(file_path,'r') as f:
            idx = 0
            for line in tqdm(f.readlines()):
                nl=line.strip()
                self.examples.append(convert_examples_to_features(nl,tokenizer,idx))
                idx = idx+1


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return torch.tensor(self.examples[i].nl_ids)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_seed()
config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer
tokenizer = tokenizer_class.from_pretrained("CS/graphcodebert/robatortoken")
model = model_class.from_pretrained("CS/graphcodebert/model").to(device)
model = Model(model, tokenizer)

eval_dataset = TextDataset(tokenizer, "evaldataset/code.txt")
eval_sampler = SequentialSampler(eval_dataset)
sourece = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=128)

# Eval!
eval_loss = 0.0
nb_eval_steps = 0

source_vecs=[]
for batch in tqdm(sourece):
    nl_inputs = batch.to(device)
    with torch.no_grad():
        nl_vec = model(nl_inputs)
        source_vecs.append(nl_vec.cpu().numpy())
    nb_eval_steps += 1
source_vecs=np.concatenate(source_vecs,0)
print(source_vecs.shape)

np.save("evaldataset/code_vecs.npy",source_vecs)

# scores=np.matmul(source_vecs,source_vecs.T)
# sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]

# with open("code_score.jsonl", 'w') as f:
#     for index,sort_id in enumerate(sort_ids):
#         js={}
#         js['idx']=index
#         js['answers']=[]
#         js["score"] = []
#         for idx in sort_id:
#             js['answers'].append(str(idx))
#             js['score'].append(str(scores[index, idx]))
#         f.write(json.dumps(js)+'\n')