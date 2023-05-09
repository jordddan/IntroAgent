# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import xlwt
from time import sleep
from tqdm import tqdm,trange
from fake_useragent import UserAgent
from urllib import parse
import wget
import os
ua=UserAgent()
TotalNum = 0

def GetInfo(url):
    chrome=ua.chrome
    head = { \
        'user-agent': chrome \
        }  # 20210607更新，防止HTTP403错误
    r = requests.get(url, headers=head)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    soup = BeautifulSoup(r.text.encode(r.encoding).decode("utf-8"), "html.parser")
    # print(soup)

    article_list = soup.find_all(class_="gs_r gs_or gs_scl")
    res = []

    for article in article_list:
        title=article.find(class_="gs_ri").find('h3').find("a").text
        url=article.find(class_="gs_or_ggsm").find("a")["href"]
        res.append(url)
        print(title," ",url)

    return res

def download_pdf(url_list,folder_path):
    index = 0
    for url in url_list:
        if "pdf" in url:
            wget.download(url=url,out=os.path.join(folder_path,f"ref{index}.pdf"))
            index += 1
        elif "arxiv" in url:
            wget.download(url=f"{url}.pdf",out=os.path.join(folder_path,f"ref{index}.pdf"))
            index += 1
        else:
            continue
           
if __name__ == '__main__':
    question="non-auto-regressive"
    question=question.strip()
    key_list=question.split(" ")
    for i in range(len(key_list)):
        key_list[i]=parse.quote(key_list[i])
    key = "+".join(key_list)
    # url = 'https://xs.dailyheadlines.cc/scholar?start=' + str(start) + '&q=' + key + '&hl=zh-CN&as_sdt=0,5'
    url="https://xueshu.dailyheadlines.cc/scholar?as_ylo=2022&hl=zh-CN&as_sdt=0%2C5&q="+key+"&btnG="
    url_list = GetInfo(url)
    folder_path = "/workspace/papergpt/openfile/ReferencePapers"
    download_pdf(url_list,folder_path)
