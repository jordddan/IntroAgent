import torch
import os
from utils.preprocess import get_data



if __name__ == "__main__":
    dataset = None
    if os.path.exists("openfile/dataset"):
        dataset = torch.load("openfile/dataset")
    else:
        dataset = get_data(4)

    
