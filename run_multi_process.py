import torch
import os
from utils.preprocess import get_data
from agent.multi_process_agent import IntroAgent
import json
if __name__ == "__main__":
    dataset = []
    # if os.path.exists("openfile/dataset"):
    #     dataset = torch.load("openfile/dataset")
    # else:
    #     dataset = get_data(8)
    
    # for item in dataset:
    #     print(item['contribution'])

    with open("data/dataset.json",'r') as f:
        data = f.readlines()
        for line in data:
            dataset.append(json.loads(line))
            
    lst = [(4,2)]

    for item in lst:
        step = item[0]

        intro_agent = IntroAgent(prompt_path=f"data/prompt.txt",dataset=dataset,batch_size=item[0],epoch=item[1],mode="rand")
        workspace = f"openfile/output/step{step}"
        if not os.path.exists(workspace):
            os.mkdir(workspace)
        print(intro_agent.ref)
        step = intro_agent.batch_size
        # intro_agent.test_all(f"{workspace}/intro_origin",600)
        # with open(f"{workspace}/prompt_original.txt",'w') as f:
        #     f.write(intro_agent.prompt)
        # intro_agent.train_multi_process()
        # with open(f"{workspace}/prompt_trained.txt",'w') as f:
        #     f.write(intro_agent.prompt)
        with open(f"{workspace}/prompt_trained.txt",'r') as f:
            prompt = f.read()
        intro_agent.prompt = prompt
        intro_agent.test_all(f"{workspace}/intro_trained",600)

    