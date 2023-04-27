from colorama import Fore, Style
import sys
import subprocess
import torch
import os
import openai
from utils.chat import single_chat, multi_chat
# 
class IntroAgent:

    def __init__(self,prompt_path,dataset):

        self.prompt_path = prompt_path
        self.dataset = dataset
        self.prompt = self.initialize_prompt()
        # import pdb
        # pdb.set_trace()
        self.n = len(dataset)

    def initialize_prompt(self):
        input = ("now you are a professor to write the introduction section of an academic paper based on the given contributions of the paper. "
                "i want you to list out the key steps to write a good introduction with the given contributions of this paper. "
                "your are only allow to generate the key steps, do not give me other irrelevant words. "
                "make sure someone can write a good introduction when follow your key steps. "
                )
        response = single_chat(input)
        with open(self.prompt_path,"w") as f:
            f.write(response)
        return response

    def forward(self,data):# F
        
        # print("\033[1;33;40mStartiing Forwad:\033[0m",end=" ")

        contribution = data["contribution"]

        p = self.prompt
        input = (f"now you are a professor to write the introduction of an academic paper of our work. "
                 f"the main contribution of the our work is {contribution}. \n"
                 f"you should follow the key steps below to write the introduction:\n "
                 f"{self.prompt}. "
                 f"you can only give me the content of the introduction section. "
                 f"do not give me any extra words. "
        )
        response = single_chat(input)
        self.prompt = p
        return response # y'

    def get_reward(self,data, predict_intro): # R
        contribution =  data["contribution"]
        introduction = data["introduction"]
        input = (f"now i want to write a introduction of an academic paper based on the contributions of our work. "
                 f"the contribution of our work is as follows {contribution} . "
                 f"now i have two versions of the introduction seciton, you should help me to pick out the better one "
                 f"the first one is:\n  {introduction}. \n \n "
                 f"the second one is:\n  {predict_intro} \n \n. "
                 f"please just tell me which one is the better one. "
                 f"you should only return one word: first or second. do not generate any other word"
        )

        response = single_chat(input)

        if "second" in response.lower():
            return None
        else:
            input_list = []
            reply_list = []
            input_list.append(input)
            input = (f"now i want to write a introduction of an academic paper based on the contributions of our work. "
                 f"the contribution of our work is as follows {contribution} . "
                 f"now i have two versions of the introduction seciton, you should help me to pick out the better one "
                 f"the first one is:\n  {introduction}. \n \n "
                 f"the second one is:\n  {predict_intro} \n \n. "
            )

            input_list.append(f"Please give me the deficiencies of the second version compared with the first version, give me advised to revise the second version")
            reply_list.append("i have recieved")
            response = multi_chat(input_list,reply_list)
            return response

    
    def backward(self,data,reward): # G

        # 在更新的时候 不能放入 contribution的内容，不然他直接把contribution的关键字抄到prompt里
        contribution = data["contribution"]
        input = (f"i have written an introduction section of an academic paper based on the contribution of our work. \n"
                 f"these are the key steps i followed to write the introduction: {self.prompt} \n"
                 f"however, the result is not very satisfying since the revier point out some shortcomings of the introduction: {reward}"
                 f"you are asked to optimize the key steps, so i can use the updatad key steps to write a better introduction"
                 f"your are only allow to generate the key steps, do not give me other irrelevant words. "
                 )
        response = single_chat(input)
        self.prompt = response
        return response
    

    def train(self):
        print("\033[1;32;40mThe Original Prompt is:\033[0m",end=" ")
        print(self.prompt)
        for i in range(self.n):
            data = self.dataset[i]
            # print(data)
            predict_intro = self.forward(data)      
            reward = self.get_reward(data,predict_intro)  
            if reward is not None:
                self.backward(data, reward)
                print("\033[1;34;40mThe Reward is:\033[0m",end=" ")
                print(reward)
                print("\033[1;33;40mThe Updated Prompt is:\033[0m",end=" ")
                print(self.prompt)