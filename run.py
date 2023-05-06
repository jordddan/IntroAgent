import torch
import os
from utils.preprocess import get_data
from agent.intro_agent import IntroAgent
from agent.icl_agent import IclAgnet
import json
if __name__ == "__main__":
    dataset = []
    # if os.path.exists("openfile/dataset"):
    #     dataset = torch.load("openfile/dataset")
    # else:
    #     dataset = get_data(8)
    
    # for item in dataset:
    #     print(item['contribution'])

    with open("openfile/dataset.json",'r') as f:
        data = f.readlines()
        for line in data:
            dataset.append(json.loads(line))

    # with open("openfile/dataset.json",'w') as f:
    #     for item in dataset:
    #         s = json.dumps(item)
    #         f.write(s+"\n")

    
    # lst = [(1,2),(2,3),(3,4),(4,5)]
    # for item in lst:
    #     intro_agent = IntroAgent(prompt_path=f"output2/step{item[0]}/prompt_origin.txt",dataset=dataset,batch_size=item[0],epoch=item[1])
    #     print(intro_agent.ref)
    #     step = intro_agent.batch_size
    #     intro_agent.test_all(f"output2/step{step}/intro_origin",600)
    #     # intro_agent.write_intro("intro_origin.txt",600)
    #     intro_agent.train()
    #     # intro_agent.write_intro("intro_trained.txt",600)
    #     intro_agent.test_all(f"output2/step{step}/intro_traianed",600)

    lst = [(1,2)]
    for item in lst:
        intro_agent = IclAgnet(prompt_path=f"icl/step{item[0]}/prompt_origin.txt",dataset=dataset,batch_size=item[0],epoch=item[1])
        print(intro_agent.ref)
        step = intro_agent.batch_size
        intro_agent.test_all(f"icl/step{step}/intro_origin",600)
        # intro_agent.write_intro("intro_origin.txt",600)
        intro_agent.train()
        # intro_agent.write_intro("intro_trained.txt",600)
        intro_agent.test_all(f"icl/step{step}/intro_traianed",600)

    