from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import os

os.environ['LANGCHAIN_PROJECT']='Sequential_llm_app'

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

llm=HuggingFaceEndpoint(
    # repo_id="HuggingFaceH4/zephyr-7b-beta",
    # repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    # repo_id="zai-org/GLM-4.7-Flash",
    repo_id="openai/gpt-oss-20b",
    # repo_id="openai/gpt-oss-120b",
    # repo_id="HuggingFaceH4/zephyr-7b-gemma-v0.1",
    # repo_id="lmsys/vicuna-13b-v1.5",
    task="text-generation"
)
# model=ChatHuggingFace(llm=llm)
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser


config={
    'tags':['llm app','report generation','summarization'],
    'metadata':{"model1":"gpt-oss-20b"}
}

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)