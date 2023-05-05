import torch
import os
from utils.preprocess import get_data
from agent.intro_agent import IntroAgent
import json
if __name__ == "__main__":
    dataset = None
    if os.path.exists("openfile/dataset"):
        dataset = torch.load("openfile/dataset")
    else:
        dataset = get_data(8)
    # for item in dataset:
    #     print(item['contribution'])


    with open("openfile/dataset.json",'w') as f:
        for item in dataset:
            s = json.dumps(item)
            f.write(s+"\n")
    
    intro_agent = IntroAgent(prompt_path="openfile/prompt.txt",dataset=dataset)
    print(intro_agent.ref)
    step = intro_agent.accumulation_step
    intro_agent.test_all(f"output/step{step}/intro_origin",600)
    # intro_agent.write_intro("intro_origin.txt",600)
    intro_agent.train()
    # intro_agent.write_intro("intro_trained.txt",600)
    intro_agent.test_all(f"output/step{step}/intro_traianed",600)
