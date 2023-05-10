import torch
from torch.utils.data import Dataset

from colorama import Fore, Style
import sys
import subprocess
import torch
import os
import openai
from utils.chat import single_chat, multi_chat
import tiktoken
import json
from random import shuffle
import copy
import vthread
from torch.utils.data import DataLoader,Dataset
# 
def collate_fn(batch):
    return batch

class DirectAgent:

    def __init__(self, dataset):
        self.dataset = dataset
        self.ori_data = copy.deepcopy(self.dataset)
        self.get_ref()
        # import pdb
        # pdb.set_trace()
        self.n = len(dataset)

    def get_ref(self):
        self.ref = ""
        for i, data in enumerate(self.dataset[:10]):
            title = data["title"]
            abstract = data["abstract"]
            line = f"{i}. {title}: \n {abstract} \n\n"
            self.ref += line
        


    @vthread.pool(20)
    def inference(self, data, words, file_name):
        # print("\033[1;33;40mStartiing Forwad:\033[0m",end=" ")

        contribution = data["contribution"]

        role = f"Now you are a professor to write the introduction section of an academic paper based on our work. "
        input = (
                 f"The main contribution of the our work is: \n \n {contribution}. "
                 f"Here are the related works, each one contain the title and the abstract:  \n{self.ref}\n"
                 f"Please write an introduction for me. \n"
                 f"The introduction should be divided into several paragraphs and total words used in this section should be more than {words}. "
                 f"You should only give me the content of the introduction section, do not give me any extra words. "
        )
        response = single_chat(input,role)

        with open(file_name,"w") as f:
                f.write(response)
        return response
    
    def test_all(self, name, words):

        for i, data in enumerate(self.ori_data):
            file_name = f"{name}{i}.txt"
            res = self.inference(data,words,file_name)
            print(i,)

        vthread.pool.wait() 
