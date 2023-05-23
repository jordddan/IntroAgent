import openai
import os
os.environ["OPENAI_API_KEY"] = 'xxxxxxxxxxx'
openai.api_type = "azure"

openai.api_base = "https://mtutor-dev.openai.azure.com/"

openai.api_version = "2023-03-15-preview"

openai.api_key = os.getenv("OPENAI_API_KEY")

def single_chat(content,role=None):

    # print("\033[1;34;40mConversation Input:\033[0m",end=" ")
    # print(content)
    if role is None:
        role = "You are an AI assistant that helps people find information."
    messages = [
                {"role":"system","content":role},
                {"role":"user","content":content}
                ]
    # try:
    #     response = openai.ChatCompletion.create(engine="mtutor-openai-dev",
    #                         messages = messages,
    #                         temperature=0,)
    # except:
    #     print(content)
    #     import pdb
    #     pdb.set_trace()
    response = openai.ChatCompletion.create(engine="mtutor-openai-dev",
                            messages = messages,
                            temperature=0,)
    res = response["choices"][0]["message"]["content"]
    # print("\033[1;32;40mConversation Response:\033[0m",end=" ")
    # print(res)

    # import pdb
    # pdb.set_trace()

    return  res

def multi_chat(input_list, reply_list, role = None):
    # len(input_list) = n, len(reply_list) = n-1
    if role is None:
        role = "You are an AI assistant that helps people find information."
    messages = [
                {"role":"system","content":role}
                ]
    for i in range(len(reply_list)):
        messages.append({"role":"user","content":input_list[i]})
        messages.append({"role":"assistant","content":reply_list[i]})
    messages.append({"role":"user","content":input_list[-1]})
    
    # print("\033[1;34;40mMuilt-Conversation Input:\033[0m",end=" ")
    # print(messages)

    response = openai.ChatCompletion.create(engine="mtutor-openai-dev",
                            messages = messages,
                            temperature=0,)
    
    res = response["choices"][0]["message"]["content"]
    # print("\033[1;32;40mMulti-Conversation Response:\033[0m",end=" ")
    # print(res)

    # import pdb
    # pdb.set_trace()

    return res
