import torch
import os
from utils.preprocess import get_data
from agent.intro_agent import IntroAgent

if __name__ == "__main__":
    dataset = None
    if os.path.exists("openfile/dataset"):
        dataset = torch.load("openfile/dataset")
    else:
        dataset = get_data(4)
    for item in dataset:
        print(item['contribution'])


    intro_agent = IntroAgent(prompt_path="openfile/ReferencePapers/prompt.txt",dataset=dataset)
    
    intro_agent.write_intro(f"intro_origin.txt")
    intro_agent.train()
    intro_agent.write_intro(f"intro_traianed.txt")
