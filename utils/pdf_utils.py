from colorama import Fore, Style
import sys


import os

import subprocess
import os

from pdfminer.high_level import extract_text

def pdf2text(filepath):
    text = extract_text(filepath)   
    output_path = filepath[:-4] + ".txt"
    # print(output_path)
    # import pdb
    # pdb.set_trace()
    with open(output_path,"w") as f:
        f.write(text)
    


if __name__ == "__main__":

    for i in range(8):
                #    openfile/ReferencePapers
        pdf2text(f"openfile/reference_papers/paper{i}.pdf")
    

