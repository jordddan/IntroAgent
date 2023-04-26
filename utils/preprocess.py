from colorama import Fore, Style
import sys


import os

import subprocess
import os

from pdfminer.high_level import extract_text
import openai
from utils.chat import single_chat
import torch

def get_data(n):
    '''
        args:
            n: means should construct data from paper 1-n 
        
        outputs:
            dataset: list of data, each element is a dict with 2 key: "introduction","contribution"
    '''
    dataset = []
    for i in range(n):
        data = {}
        with open(f"openfile/ReferencePapers/intro{i}.txt") as f:
            intro = f.read()
        input = ("here is the introduction sectionof an academic paper,"
                "please read it carefully and extract the main contribution of this paper")
        response = single_chat(input)
        data["introduction"] = intro
        data["contribution"] = response
        dataset.append(data)
    torch.save(dataset,"openfile/dataset")
    return dataset
      

    