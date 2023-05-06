# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.text_splitter import CharacterTextSplitter
# from langchain import OpenAI,VectorDBQA
# from langchain.document_loaders import DirectoryLoader
# from langchain.chains import RetrievalQA

# # 加载文件夹中的所有txt类型的文件
# loader = DirectoryLoader('/content/sample_data/data/', glob='**/*.txt')
# # 将数据转成 document 对象，每个文件会作为一个 document
# documents = loader.load()

# # 初始化加载器
# text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# # 切割加载的 document
# split_docs = text_splitter.split_documents(documents)

# # 初始化 openai 的 embeddings 对象
# embeddings = OpenAIEmbeddings()
# # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
# docsearch = Chroma.from_documents(split_docs, embeddings)

# # 创建问答对象
# qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch,return_source_documents=True)
# # 进行问答
# result = qa({"query": "科大讯飞今年第一季度收入是多少？"})
# print(result)

import openai
import os
os.environ["OPENAI_API_KEY"] = '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 '

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

x = "today is a good day"

ans = get_embedding(x)

import json 
import torch
import torch.nn as nn

with open('openfile/dataset.json','r') as f:
   data = f.readlines()

emb = []

for i, string in enumerate(data):
   line = json.loads(string)
   contribution = line["contribution"]
   embedding = get_embedding(contribution)
   emb.append(embedding)


emb = torch.tensor(emb)
print(emb.shape)

n,c = emb.shape[0],emb.shape[1]
cos = nn.CosineSimilarity(dim=-1)
sim = cos(emb.unsqueeze(1),emb.unsqueeze(0))

_, res = torch.sort(sim,dim=-1,descending=True)
print(res)

new_data = []

for i in range(n):
    sim_index = [res[i][1].item(),res[i][2].item()]
    
    line = data[i]
    line_dict = json.loads(data[i])
    line_dict["sim_idx"] = sim_index
    line = json.dumps(line_dict)
    new_data.append(line)

with open("openfile/dataset.json",'w') as f:
    for data in new_data:
       f.write(data+"\n")
    



    
