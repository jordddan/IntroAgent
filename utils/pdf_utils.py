from colorama import Fore, Style
import sys


import os

import subprocess
import os
import wget
from pdfminer.high_level import extract_text

def pdf2text(filepath):
    text = extract_text(filepath)   
    output_path = filepath[:-4] + ".txt"
    # print(output_path)
    # import pdb
    # pdb.set_trace()
    with open(output_path,"w") as f:
        f.write(text)

def deduplicate(url_list):
    st = set(url_list)
    st = list(st)
    return st

def download_pdf(url_list,path):

    url_list = deduplicate(url_list)
    import pdb
    pdb.set_trace()
    for i, url in enumerate(url_list):
        file_name = f"paper{i}.pdf"
        try:
            wget.download(url=url, out=os.path.join(path,file_name))
        except:   
            import pdb
            pdb.set_trace()
        

if __name__ == "__main__":

    # for i in range(8):
    #             #    openfile/ReferencePapers
    #     pdf2text(f"openfile/reference_papers/paper{i}.pdf")

    with open("openfile/pdf_link.txt",'r') as f:
        url_list = f.readlines()
    download_pdf(url_list,"openfile/pdf_file")
        

