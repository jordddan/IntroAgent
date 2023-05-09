from colorama import Fore, Style
import sys


import os

import subprocess
import os

from pdfminer.high_level import extract_text
import openai
import torch
import json
from get_paper_from_pdf import Paper
import tiktoken
from chat import single_chat

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


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
        # with open(f"openfile/ReferencePapers/intro{i}.txt") as f:
        #     intro = f.read()
        # input = (f"here is the introduction section of an academic paper: {intro} "
        #         "please read it carefully and extract the main contributions of this paper")
        # response = single_chat(input)
        print(i)
        path = f"openfile/reference_papers/paper{i}.pdf"
        paper = Paper(path=path)
        paper.parse_pdf()
        print(paper.section_names)
        if i in [2,5,6]:

            with open(f"openfile/reference_papers/intro{i}.txt", "r") as f:
                intro = f.read()
            with open(f"openfile/reference_papers/abstract{i}.txt", "r") as f:
                abstract = f.read()
        else:
            intro = paper.section_texts["Introduction"]
            abstract = paper.section_texts["Abstract"]
        if i == 6:
            title = "IMPROVING NON-AUTOREGRESSIVE TRANSLATION MODELS WITHOUT DISTILLATION"
        else:
            title = paper.title

        input = (f"here is the introduction section of an academic paper: {intro} "
                f"please read it carefully and extract the main contributions of this paper."
                f"the contribution should be as detailed as possible."
                f"you should only return the extracted contribution without any prompt word.")
        response = single_chat(input)

        words = num_tokens_from_string(intro,"cl100k_base")
        data["title"] = title
        data["abstract"] = abstract
        data["intro_words"] = words
        data["introduction"] = intro
        data["contribution"] = response

        dataset.append(data)
    torch.save(dataset,"openfile/dataset")
    return dataset
      
def extract_intro(n):
    '''
        args:
            n: means should construct data from paper 1-n 
        
        outputs:
            dataset: list of data, each element is a dict with 2 key: "introduction","contribution"
    '''

    for i in range(0,n):
        data = {}
        # with open(f"openfile/ReferencePapers/intro{i}.txt") as f:
        #     intro = f.read()
        # input = (f"here is the introduction section of an academic paper: {intro} "
        #         "please read it carefully and extract the main contributions of this paper")
        # response = single_chat(input)
        print(i)
        path = f"openfile/data/paper{i}.pdf"
        text_path = f"openfile/data/paper{i}.txt"
        intro_path = f"openfile/data/intro{i}.txt"
        abstract_path = f"openfile/data/abstract{i}.txt"
        title_path = f"openfile/data/title{i}.txt"
        paper = Paper(path=path)
        paper.parse_pdf()
        all_text = extract_text(path)
        with open(text_path,'w') as f:
            f.write(all_text)
        print(paper.section_names)
        intro = ""
        abstract = ""
        if "Introduction" in paper.section_texts:
            intro = paper.section_texts["Introduction"]
        if "Abstract" in paper.section_texts:
            abstract = paper.section_texts["Abstract"]
        title = paper.title 

        # with open(intro_path,'w') as f:
        #     f.write(intro)

        with open(abstract_path,'w') as f:
            f.write(abstract)

        # with open(title_path,'w') as f:
        #     f.write(title)

def get_dataset(n):

    dataset = []
    for i in range(n):
        with open(f"openfile/data/title{i}.txt", 'r') as f:
            title = f.read()
        with open(f"openfile/data/intro{i}.txt", 'r') as f:
            intro = f.read()
        with open(f"openfile/data/abstract{i}.txt", 'r') as f:
            abstract = f.read()
        
        words = num_tokens_from_string(intro,"cl100k_base")

        input = (f"Here is the introduction section of an academic paper: {intro} "
                f"please read it carefully and extract the main contributions of this paper."
                f"The contribution should be as detailed as possible."
                f"you should only return the extracted contribution without any prompt word.")
        response = single_chat(input)
        contribution = response

        data = dict()
        data["introduction"] = intro
        data["contribution"] = contribution
        data["title"] = title 
        data["words"] = words
        data["abstract"] = abstract

        dataset.append(data)
        print(i,title)
    with open("openfile/dataset.json",'w') as f:
        for data in dataset:
            f.write(json.dumps(data)+"\n")

if __name__ == "__main__":
    # extract_intro(73)
    get_dataset(73)