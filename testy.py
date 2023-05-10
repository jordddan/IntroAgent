
import torch
import os
from utils.preprocess import get_data
from agent.intro_agent import IntroAgent
from agent.icl_agent import IclAgnet
import json
from utils.chat import single_chat
import vthread

@vthread.pool(20) # 只用加这一行就能实现6条线程池的包装
def compare_text(text1,text2,score,index,dataset,global_index):
    contribution = dataset[global_index]["contribution"]
    role = (f"You are a professor, "
            f"you know how to write a good introduction of an academic paper based on the contribution of the paper. ") 
    print(len(text1))
    print(len(text2))
    input = (
             f"Now i have write two different version of introduction section for my paper. "
             f"The contribution of the paper is {contribution}"
             f"I need you to decide which one is written better as an introduction section of my paper."
             f"Here is the first one: \n\n {text1} \n\n"
             f"Here is the second one: \n\n {text2} \n\n"
             f"You should make comparisons from the perspective of academic paper writing and select the better one."
             f"If you think the first one is better, return only one word 'first'. "
             f"If you think the second one is better, return only one word 'second'. "
             f"Your answer must be one of the words 'first' or 'second', do not give me any other word. "
             )
    
    response = single_chat(input,role)
    # import pdb
    # pdb.set_trace()
    print(response)
    if 'first' in response.lower():
        score[index] += 1
    
if __name__ == "__main__":
    
    intro_direct = []
    intro_prompt_direct = []
    intro_prompt_trained = []
    intro_expert_direct = []
    intro_expert_trained = []

    n = 72

    dataset = []
    with open("data/dataset.json",'r') as f:
        data = f.readlines()
        for line in data:
            dataset.append(json.loads(line))
    

    for i in range(n):
        file_path = f"openfile/direct/intro_origin{i}.txt"
        with open(file_path,"r") as f:
            intro = f.read()
            if len(intro) == 0:
                raise "the introduction file is empty"
            intro_direct.append(intro)

        file_path = f"openfile/output/step8/intro_origin{i}.txt"
        with open(file_path,"r") as f:
            intro = f.read()
            if len(intro) == 0:
                raise "the introduction file is empty"
            intro_prompt_direct.append(intro)
        
        file_path = f"openfile/output/step8/intro_trained{i}.txt"
        with open(file_path,"r") as f:
            intro = f.read()
            if len(intro) == 0:
                raise "the introduction file is empty"
            intro_prompt_trained.append(intro)

        file_path = f"openfile/output_expert/step8/intro_origin{i}.txt"
        with open(file_path,"r") as f:
            intro = f.read()
            if len(intro) == 0:
                raise "the introduction file is empty"
            intro_expert_direct.append(intro)

        file_path = f"openfile/output_expert/step8/intro_trained{i}.txt"
        with open(file_path,"r") as f:
            intro = f.read()
            if len(intro) == 0:
                raise "the introduction file is empty"
            intro_expert_trained.append(intro)

    data_all = [intro_direct, intro_prompt_direct, intro_prompt_trained, intro_expert_direct, intro_expert_trained]
    scores = [0 for i in range(5)]

    for i in range(n):
        score = [0 for i in range(5)]
        for ii in range(5):
            print(ii)
            for jj in range(5):
                if ii != jj:
                    text1 = data_all[ii][i]
                    text2 = data_all[jj][i]
                    compare_text(text1, text2, score, ii, dataset, i)
            
        vthread.pool.wait()   
        for i in range(5):
            scores[i] += score[i]

    print(scores)




    [62, 28, 28, 51, 34]



