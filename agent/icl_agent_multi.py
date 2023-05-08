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
# 


class IclAgnetMulti:

    def __init__(self, prompt_path, dataset, batch_size, epoch):

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
        
    def get_ref(self):
        self.ref = ""
        for i, data in enumerate(self.dataset):
            title = data["title"]
            abstract = data["abstract"]
            line = f"{i}. {title}: \n {abstract} \n\n"
            self.ref += line
        

    def initialize_prompt(self):

        '''
            Please include five paragraph: Establishing the motivation for the research. Explaining its importance and relevance to the AI community. Clearly state the problem you're addressing, your proposed solution, and the specific research questions or objectives. Briefly mention key related work for context. Explain the main differences from your work. 
        '''
        # role = "now you are a professor to give guidance on writing the introduction section of an machine learning paper based on the given contributions of this work "
        # input = (
        #         "i want you to make a guidance to write a good introduction of a machine learning paper. "
        #         "you are only allow to generate the guidance, do not give me other irrelevant words. "
        #         "make sure someone can write a good introduction when follow your guidance. "
        #         )
        # response = single_chat(input,role)

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

    def construct_icl(self,data):
        idxs = data["sim_idx"]

        icl1 = self.ori_data[idxs[0]]
        contribution1 = icl1["contribution"]
        introduction1 = icl1["introduction"]

        icl2 = self.ori_data[idxs[1]]
        contribution2 = icl2["contribution"]
        introduction2 = icl2["introduction"]

        prompt = (
                  f"Here are some examples of writing an introduction section following the given key steps \n"
                  f"Example 1: \n\n Contribution: {contribution1} \n Introduction:{introduction1}\n " 
                  f"Example 2: \n\n Contribution: {contribution2} \n Introduction:{introduction2}\n "
                  )
        return prompt 

            
    def forward(self,data):# F
        
        # print("\033[1;33;40mStartiing Forwad:\033[0m",end=" ")

        contribution = data["contribution"]
        words = data["intro_words"]
        p = self.prompt
        icl_context = self.construct_icl(data)
        role = (f"I need you to write an introduction section of an academic paper following the given steps. "
                f"The introduction section should base on given contribution.")
        input1 = (
                #  f"The main contribution of the our work is: \n \n {contribution}. "
                 f"This is the key steps to write a good introduction section: \n \n {self.prompt} \n \n {icl_context}"
                 f"You can read them and learn how to follow these steps to write a good introduction based on the contributions. "
        )
        reply = ("I have read them, i will follow the key steps to write an introduction section, please give me the "
                 "please provide me the contribution of the paper.")
        input2 = (
                f"Here are the contributions of the introduction section you are going to write: \n\n {contribution}. \n"
                f"Here are the related works about the accademic paper, each one contain the title and the abstract:  \n{self.ref}\n"
                f"You should follow the key steps above to write the introduction section based on the contribution and related works given above:\n \n"
                f"The introduction should be divided into several paragraphs and total words used in this section should be more than {words}. "
                f"You should only give me the content of the introduction section, do not give me any extra words. "
                f"When you write the introduction, make sure the content and the order of the paragraphs strictly follow the key steps. "
        )
      
        input_list = [input1,input2]
        reply_list = [reply]

        response = multi_chat(input_list,reply_list,role)
        self.prompt = p
        return response # y''

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

            # role = ("You are a professor to read proc")
            # input = (
            #      f"You will receive two difference introductions of the same academical paper of "
            #      f"and you should read them carefully and make comparisons"
            #      f"The main contribution of the paper is as follows: \n {contribution} . \n\n "
            #      f"The first version is: \n\n {predict_intro} \n\n "
            #      f"The second version is: \n\n {introduction} "
            # )

            # role = ("You are a professor to write an introduction section of an academic paper of machine learning. ")
            # input = (
            #      f"The main contribution of the paper is as follows: \n {contribution} . \n\n "
            #      f"You will receive two difference versions of the introduction section. "
            #      f"You should read both of them carefully and make comparisons to tell me which one is better as an introduction section of an academic paper."
            #      f"The first version is: \n\n {predict_intro} \n\n "
            #      f"The second version is: \n\n {introduction} \n\n "
            #     #  f"Note that the paper is an academic paper, the readers are likely to be experts. "
            # )

            role = ("You are a professor of machine learning, and you are guiding your student on how to write "
                    "an introduction section of a machine learning paper, based on the contribution. ")

            input = (
                 f"The main contribution of the paper is as follows: \n {contribution} . \n\n "
                 f"This is the introduction your student write: \n\n {predict_intro} \n\n"
                 f"This is the correct answer: \n\n {introduction} \n\n "
                 f"You need to point out the shortcomings in the introduction written by "
                 f"your student compared with the correct answer.  "
                 f"You should read both of them carefully and make comparisons from the perspective of academical paper writing. "
                 f"For example, you can make considerations like: Which paragraph should you write first? "
                 f"Is the logic of the sections coherent? and so on. "
            )
            
            response = single_chat(input,role)

            return response

    
    def backward(self,data,reward): # G

        # 在更新的时候 不能放入 contribution的内容，不然他直接把contribution的关键字抄到prompt里
        contribution = data["contribution"]
        introduction = data["introduction"]
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
                 f"5. Just give me the revised key steps, do not generate other words. \n\n"
                 f"You can give me 3 candidate revised version of the key steps, seperated by '***'. ")
        response = single_chat(input,role)
        
        candidate = response

        role =  (f"You are a professor who can give a guidance of how to write a good introduction section of an academic paper")
        input = (
                 f"Writing a good introduction section should follow a correct guidance (key steps). "
                 f"Here are three candidate version of the guidance: \n {candidate} \n "
                 f"Here is an introduction \n {introduction} \n\n. "
                 f"It is written following one of the guidance above. "
                 f"I would like to know which version of the guidance have the same structure with the given introduction. "
                 f"I want you to pick out a guidance for me, so i can write the above introduction following this guidance. "
                 f"Do not only give me the number of the candidate, please provide me with the complete content of the picked out guidance. "
                )
        
        response = single_chat(input,role)


        self.prompt = response
        return response
    
    def gradient_accumulation(self, rewards):

        role = "You are a review summary robot that can help me summarize the reviewers' feedback."
        input = (
                 f"I have written a machine learning paper, here are some review comments from different reviewers: " 
                 f"\n\n {rewards} \n\n"
                 f"You should help me conclude the common important comments, make sure there are no repetitions. \n"
                 f"You should summerize these comments to the most important 1 or 2 points, i just want the most important one or two advice. \n"
                 f"Your response should only contain the summarized reviews. "
                 )
        response = single_chat(content=input, role=role)

        return response

    def train(self):
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
            for i in range(self.n):
                data = self.dataset[i]
                predict_intro = self.forward(data)      
                reward = self.get_reward(data,predict_intro)  

                if reward is not None:
                    accumulate_rewards += f"review comments{i}: {reward} \n\n "
                else:
                    self.st.add(i)
                    if len(self.st) > 3:
                        break
                    
                # gradient accumulation update
                if (i+1) % self.batch_size == 0 or i==7:
                    accumulate_rewards = self.gradient_accumulation(accumulate_rewards)
                    print("\033[1;34;40mThe Reward is:\033[0m",end=" ")
                    print(accumulate_rewards)
                    self.backward(data, accumulate_rewards)
                    accumulate_rewards = ""
                    print("\033[1;33;40mThe Updated Prompt is:\033[0m",end=" ")
                    print(self.prompt)

            print(self.st)
            if len(self.st) > 3:
                break

        with open(f"openfile/iclmulti/step{self.batch_size}/prompt_trained.txt",'w') as f:
            f.write(self.prompt)

    def write_intro(self,file_path, words):
        data = self.dataset[0]
        intro = self.inference(data, words)
        with open(file_path,'w') as f:
            f.write(intro)


    def inference(self, data, words):
        # print("\033[1;33;40mStartiing Forwad:\033[0m",end=" ")

        contribution = data["contribution"]
        words = data["intro_words"]
        p = self.prompt
        icl_context = self.construct_icl(data)
        role = (f"I need you to write an introduction section of an academic paper following the given steps. "
                f"The introduction section should base on given contribution.")
        input1 = (
                #  f"The main contribution of the our work is: \n \n {contribution}. "
                 f"This is the key steps to write a good introduction section: \n \n {self.prompt} \n \n {icl_context}"
                 f"You can read them and learn how to follow these steps to write a good introduction based on the contributions. "
        )
        reply = ("I have read them, i will follow the key steps to write an introduction section, please give me the "
                 "please provide me the contribution of the paper.")
        input2 = (
                f"Here are the contributions of the introduction section you are going to write: \n\n {contribution}. \n"
                f"Here are the related works about the accademic paper, each one contain the title and the abstract:  \n{self.ref}\n"
                f"You should follow the key steps above to write the introduction section based on the contribution and related works given above:\n \n"
                f"The introduction should be divided into several paragraphs and total words used in this section should be more than {words}. "
                f"You should only give me the content of the introduction section, do not give me any extra words. "
                f"When you write the introduction, make sure the content and the order of the paragraphs strictly follow the key steps. "
        )
      
        input_list = [input1,input2]
        reply_list = [reply]

        response = multi_chat(input_list,reply_list,role)
        self.prompt = p
        return response # y''
 
    def test_all(self, name, words):

        for i, data in enumerate(self.ori_data):
            intro = self.inference(data,words)
            file_name = f"{name}{i}.txt"
            with open(file_name,"w") as f:
                f.write(intro)
