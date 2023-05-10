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

class IntroAgent:

    def __init__(self, prompt_path, dataset, batch_size, epoch, mode="rand"):

        self.mode = mode
        self.prompt_path = prompt_path
        self.dataset = dataset
        self.ori_data = copy.deepcopy(self.dataset)
        self.prompt = self.initialize_prompt()
        self.get_ref()
        # import pdb
        # pdb.set_trace()
        self.n = len(dataset)
        self.batch_size = batch_size
        self.epoch = epoch
        self.dataloader = DataLoader(dataset,batch_size=self.batch_size,collate_fn=collate_fn)

    def get_ref(self):
        self.ref = ""
        for i, data in enumerate(self.dataset[:10]):
            title = data["title"]
            abstract = data["abstract"]
            line = f"{i}. {title}: \n {abstract} \n\n"
            self.ref += line
        

    def initialize_prompt(self):

        '''
            Please include five paragraph: Establishing the motivation for the research. Explaining its importance and relevance to the AI community. Clearly state the problem you're addressing, your proposed solution, and the specific research questions or objectives. Briefly mention key related work for context. Explain the main differences from your work. 
        '''
        if self.mode == "rand":

            role = "now you are a professor to give guidance on writing the introduction section of an machine learning paper based on the given contributions of this work "
            input = (
                    "i want you to make a guidance to write a good introduction of a machine learning paper. "
                    "you are only allow to generate the guidance, do not give me other irrelevant words. "
                    "make sure someone can write a good introduction when follow your guidance. "
                    )
            response = single_chat(input,role)

        elif self.mode == "expert":

            response = (
                        "Establishing the motivation for the research. Explaining its importance and relevance to the AI community. "
                        "Clearly state the problem you're addressing, your proposed solution,"
                        "and the specific research questions or objectives. Briefly mention key related work for context."
                        "Explain the main differences from your work."
                        )
        print(self.prompt_path)
        with open(self.prompt_path,"w") as f:
            f.write(response)
        return response

    def write(self,data):# F
        
        # print("\033[1;33;40mStartiing Forwad:\033[0m",end=" ")

        contribution = data["contribution"]
        words = data["words"]
        p = self.prompt
        role = f"Now you are a professor to write the introduction section of an academic paper based on our work. "
        input = (
                 f"The main contribution of the our work is: \n \n {contribution}. "
                 f"Here are the related works, each one contain the title and the abstract:  \n{self.ref}\n"
                 f"You should follow the key steps below to write the introduction section based on the contribution and related works given above:\n \n"
                 f"{self.prompt}. "
                 f"The introduction should be divided into several paragraphs and total words used in this section should be more than {words}. "
                 f"You should only give me the content of the introduction section, do not give me any extra words. "
        )
        response = single_chat(input,role)
        self.prompt = p
        return response # y'
        
    def get_reward(self,data, predict_intro): # R
        contribution =  data["contribution"]
        introduction = data["introduction"]
        role = f"Now you are a professor who can select the better introdcution for a machine learning paper"
        input = (
                 f"The contribution of the paper is as follows {contribution} . "
                 f"Now i have two versions of the introduction seciton, you should help me to pick out the better one "
                 f"The first one is:\n  {introduction}. \n \n "
                 f"The second one is:\n  {predict_intro} \n \n. "
                 f"Please just tell me which one is the better one. "
                 f"You should only return one word: 'first' or 'second'. do not generate any other word."
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

            role = ("You are a professor of machine learning, and you are guiding your student on how to write "
                    "an introduction section of a machine learning paper, based on the contribution. ")

            input = (
                 f"The main contribution of the paper is as follows: \n {contribution} . \n\n "
                 f"This is the introduction your student write: \n\n {predict_intro} \n\n"
                 f"This is the correct answer: \n\n {introduction} \n\n "
                 f"You need to point out the shortcomings in the introduction written by "
                 f"your student compared with the correct answer.  "
                 f"You should read both of them carefully and make comparisons from the perspective of academical paper writing. "
                 f"For example you can make considerations like: Is the order of paragraphs correct? Is the logic of the sections coherent? and so on. "
                 f"Your answer should not exceeds 100 words. "
            )
            
            response = single_chat(input,role)

            return response

    
    def backward(self,reward): # G

        # 在更新的时候 不能放入 contribution的内容，不然他直接把contribution的关键字抄到prompt里

        role =  ("You are a robot that provides academic writing key steps for introduction seciton "
                ", and others write introduction section of an academic paper based on the key steps you provide. ")
        input = (f"A student have written an introduction section of a machine learning paper based on your key steps"
                 f"Here is the original key steps you have provided: \n \n {self.prompt} \n"
                 f"However, some reviewers point out some shortcomings of the introduction: \n \n {reward} \n \n"
                 f"You should revise the original key steps by considering the reviewer's review, "
                 f"so the paper written according to the new key steps will not have the shortcomings mentioned above. \n\n"
                 f"Here are some instructions for you to revise the original steps: "
                 f"1. If you think that the shortcomings raised in the review have already been addressed in the key steps, then you can choose not to modify those steps. \n"
                 f"2. Make sure there is no repeated or similar steps in the key steps. \n"
                 f"3. You can also rearrange the order of the steps to make the introduction smoother when written according to these steps. \n"
                 f"4. There should be no more than 6 steps in total, please carefully organize the steps. \n" 
                 f"5. Just give me the revised key steps, do not generate other words. \n")
        response = single_chat(input,role)
        self.prompt = response
        return response
    
    def gradient_accumulation(self, rewards):

        role = "You are a review summary robot that can help me summarize the reviewers' feedback."
        input = (f"I have written a machine learning paper, here are some review comments from different reviewers: " 
                 f"\n\n {rewards} \n\n"
                 f"You should help me conclude the common important comments, make sure there are no repetitions. "
                 f"You should summerize these comments to the most important one or two points from these comments. "
                 f"Your response should only contain the summarized reviews."
                 f"Your response cannot exceed 50 words")
        response = single_chat(content=input, role=role)

        return response

    @vthread.pool(8)
    def forward(self, data, res):
        intro = self.write(data)
        reward = self.get_reward(data,intro)
        # print(reward)
        res.append(reward)

    def train_multi_process(self):
        print("\033[1;32;40mThe Original Prompt is:\033[0m",end=" ")
        print(self.prompt)
        flag = False
        cnt = 0
        while True:
            cnt += 1
            self.st = set()
            if cnt > self.epoch:
                break
            accumulate_rewards = ""
            shuffle(self.dataset)
            for i,batch in enumerate(self.dataloader):
                rewards_list = []
                for data in batch:
                    self.forward(data,rewards_list)
                vthread.pool.wait() 

                # import pdb
                # pdb.set_trace()
                text = ""
                for i, review in enumerate(rewards_list):
                    if review is not None:
                        text += f"Review {i}: \n\n {review} \n\n"
                accumulate_rewards = self.gradient_accumulation(text)
                print("\033[1;34;40mThe Reward is:\033[0m",end=" ")
                print(accumulate_rewards)

                self.backward(accumulate_rewards)

                print("\033[1;33;40mThe Updated Prompt is:\033[0m",end=" ")
                print(self.prompt)
                # import pdb
                # pdb.set_trace()

    def write_intro(self,file_path, words):
        data = self.dataset[0]
        intro = self.inference(data, words)
        with open(file_path,'w') as f:
            f.write(intro)

    @vthread.pool(20)
    def inference(self, data, words, file_name):
        # print("\033[1;33;40mStartiing Forwad:\033[0m",end=" ")

        contribution = data["contribution"]

        p = self.prompt
        role = f"Now you are a professor to write the introduction section of an academic paper based on our work. "
        input = (
                 f"The main contribution of the our work is: \n \n {contribution}. "
                 f"Here are the related works, each one contain the title and the abstract:  \n{self.ref}\n"
                 f"You should follow the key steps below to write the introduction section based on the contribution and related works given above:\n \n"
                 f"{self.prompt}. "
                 f"The introduction should be divided into several paragraphs and total words used in this section should be more than {words}. "
                 f"You should only give me the content of the introduction section, do not give me any extra words. "
        )
        response = single_chat(input,role)
        self.prompt = p
        with open(file_name,"w") as f:
                f.write(response)

 
    def test_all(self, name, words):

        for i, data in enumerate(self.ori_data):
            file_name = f"{name}{i}.txt"
            self.inference(data,words,file_name)

        vthread.pool.wait() 
