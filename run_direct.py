import torch
import os
from utils.preprocess import get_data
from agent.direct_agent import DirectAgent
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
            

        
    intro_agent = DirectAgent(dataset=dataset)
    workspace = f"openfile/direct"
    if not os.path.exists(workspace):
        os.mkdir(workspace)
    print(intro_agent.ref)
    intro_agent.test_all(f"{workspace}/intro_origin",600)

    