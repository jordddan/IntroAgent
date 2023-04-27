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
        role = "now you are a professor to give guidance on write the introduction section of an academic paper based on the given contributions of the paper. "
        input = (
                "i want you make a guidance to write a good introduction based on the awareness of this paper. "
                "your are only allow to generate the guidance, do not give me other irrelevant words. "
                "make sure someone can write a good introduction when follow your guidance. "
                )
        response = single_chat(input,role)
        with open(self.prompt_path,"w") as f:
            f.write(response)
        return response

    def forward(self,data):# F
        
        # print("\033[1;33;40mStartiing Forwad:\033[0m",end=" ")

        contribution = data["contribution"]

        p = self.prompt
        role = f"now you are a professor to write the introduction section of an academic paper of our work. "
        input = (
                 f"the main contribution of the our work is {contribution}. \n"
                 f"you should follow the guidance below to write the introduction:\n "
                 f"{self.prompt}. "
                 f"you can only give me the content of the introduction section. "
                 f"do not give me any extra words. "
        )
        response = single_chat(input,role)
        self.prompt = p
        return response # y'

    def get_reward(self,data, predict_intro): # R
        contribution =  data["contribution"]
        introduction = data["introduction"]
        role = f" now you are a professor who can select the better introdcution who is suitable for an academic paper"
        input = (
                 f"the contribution of the paper is as follows {contribution} . "
                 f"now i have two versions of the introduction seciton, you should help me to pick out the better one "
                 f"the first one is:\n  {introduction}. \n \n "
                 f"the second one is:\n  {predict_intro} \n \n. "
                 f"please just tell me which one is the better one. "
                 f"you should only return one word: first or second. do not generate any other word"
        )

        response = single_chat(input,role)

        if "second" in response.lower():
            return None
        else:
            # print("\033[1;33;40mPredict Intro:\033[0m",end=" ")
            # print(predict_intro)

            input_list = []
            reply_list = []
            input_list.append(input)
            role = (f"you will receive two difference introductions of the same academic paper alone with the contribution of the paper."
                    f"and you should read them carefully and make comparisons")
            input = (
                 f"the contribution of the paper is as follows: \n {contribution} . \n\n "
                 f"the first version is: {predict_intro} \n\n "
                 f"the second version is: {introduction} \n\n "
            )

            input_list.append(f"Please give me the deficiencies of the first version compared with the second version, give me advised to revise the first version for the better")
            reply_list.append("i have recieved")
            response = multi_chat(input_list, reply_list, role)
            return response

    
    def backward(self,data,reward): # G

        # 在更新的时候 不能放入 contribution的内容，不然他直接把contribution的关键字抄到prompt里
        contribution = data["contribution"]
        input = (f"i have written an introduction section of an academic paper based on the contribution of our work. \n"
                 f"these are the guidance i followed to write the introduction: {self.prompt} \n"
                 f"however, the result is not very satisfying since the reviewer point out some shortcomings of the introduction: {reward}"
                 f"you are asked to optimize the guidance, so i can use the updatad guidance to write a better introduction"
                 f"your are only allow to generate the key steps, do not give me other irrelevant words. "
                 f"make sure there is no duplicate content in the guidance"
                 f"make sure the works in the guidance do not exceed 300")
        response = single_chat(input)
        self.prompt = response
        return response
    

    def train(self):
        print("\033[1;32;40mThe Original Prompt is:\033[0m",end=" ")
        print(self.prompt)
        flag = False
        cnt = 0
        while True:
            cnt += 1
            if cnt > 3:
                break
            for i in range(self.n):
                
                data = self.dataset[2]
                # print(data)
                predict_intro = self.forward(data)      
                reward = self.get_reward(data,predict_intro)  
                if reward is not None:
                    self.backward(data, reward)
                    print("\033[1;34;40mThe Reward is:\033[0m",end=" ")
                    print(reward)
                    print("\033[1;33;40mThe Updated Prompt is:\033[0m",end=" ")
                    print(self.prompt)
                else:
                    flag = True
                    break
            if flag:
                break

    def write_intro(self,file_path):
        data = self.dataset[3]
        intro = self.forward(data)
        with open(file_path,'w') as f:
            f.write(intro)



