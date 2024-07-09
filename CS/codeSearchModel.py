# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

class Model(nn.Module):   
    def __init__(self, encoder,tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tokenizer=tokenizer
    
        
    def forward(self, code_inputs,nl_inputs=None):
        bs=code_inputs.shape[0]
        if isinstance(nl_inputs,list):
            l = len(nl_inputs)
            inputs=torch.cat([code_inputs]+nl_inputs,0)
            outputs=self.encoder(inputs,attention_mask=inputs.ne(1))[1]
            code_vec=outputs[:bs]
            nl_vecs=[outputs[bs*(i+1):bs*(i+2)]  for i in range(l)]
            return code_vec,nl_vecs
        elif nl_inputs != None:
            inputs=torch.cat((code_inputs,nl_inputs),0)
            outputs=self.encoder(inputs,attention_mask=inputs.ne(1))[1]
            code_vec=outputs[:bs]
            nl_vec=outputs[bs:]
            return code_vec,nl_vec
        else:
            outputs=self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[1]
            return outputs
        


