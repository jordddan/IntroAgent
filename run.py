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
    
    lst = [(1,2),(2,3),(3,4),(4,5)]
    for item in lst:
        intro_agent = IntroAgent(prompt_path=f"output2/step{item[0]}/prompt_origin.txt",dataset=dataset,batch_size=item[0],epoch=item[1])
        print(intro_agent.ref)
        step = intro_agent.batch_size
        intro_agent.test_all(f"output2/step{step}/intro_origin",600)
        # intro_agent.write_intro("intro_origin.txt",600)
        intro_agent.train()
        # intro_agent.write_intro("intro_trained.txt",600)
        intro_agent.test_all(f"output2/step{step}/intro_traianed",600)
