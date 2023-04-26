import openai
import os
os.environ["OPENAI_API_KEY"] = 'a41acec784d340b184bc71d850c97a7f'
openai.api_type = "azure"

openai.api_base = "https://mtutor-dev.openai.azure.com/"

openai.api_version = "2023-03-15-preview"

openai.api_key = os.getenv("OPENAI_API_KEY")

def single_chat(content):
    messages = [
                {"role":"system","content":"You are an AI assistant that helps people find information."},
                {"role":"user","content":content}
                ]
    response = openai.ChatCompletion.create(engine="mtutor-openai-dev",
                            messages = messages,
                            temperature=0,)
    return response["choices"][0]["message"]["content"],response["choices"][0]["message"]["role"]