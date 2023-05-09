
import torch
import os
from utils.preprocess import get_data
from agent.intro_agent import IntroAgent
from agent.icl_agent import IclAgnet
import json
from utils.chat import single_chat

def compare_text(text1,text2):

    role = (f"You are a peer reviewer for an academic paper, "
            f"and you are able to compare two paragraphs and determine which one is better from the perspective of an academic paper. ") 
    input = (f"Now there are two different introduction of an academic paper. "
             f"Here is the first one: \n\n {text1} \n\n"
             f"Here is the second one: \n\n {text2} \n\n"
             f"I want you to tell me which one is better. "
             f"The answer should be one word. "
             f"If you think the first one is better, return 'first'. "
             f"If you think the second one is better, return  'second'. "
             )
    
    response = single_chat(input,role)
    # import pdb
    # pdb.set_trace()
    print(response)
    if 'first' in response.lower():
        return True
    elif 'second' in response.lower():
        return False
    else:
        import pdb
        pdb.set_trace()
    
if __name__ == "__main__":
    workspace = "openfile/output/step4"
    cnt1 = 0
    cnt2 = 0
    for i in range(8):
        origin_file = f"{workspace}/intro_origin{i}.txt"
        trained_file = f"{workspace}/intro_traianed{i}.txt"
        origin_intro = ""
        trained_intro = ""

        with open(origin_file,"r") as f:
            origin_intro = f.read()
        with open(trained_file,"r") as f:
            trained_intro = f.read()
        
        flag1 = compare_text(trained_intro,origin_intro)
        flag2 = compare_text(origin_intro,trained_intro)
        if flag1:
            cnt1 += 1
        if flag2:
            cnt2 += 1
    print(cnt1)
    print(cnt2)
        



