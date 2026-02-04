
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
import numpy as np

from dotenv import load_dotenv

from langsmith import traceable # <---traceable

load_dotenv()
import os
os.environ['LANGCHAIN_PROJECT']='Rag'

from PyPDF2 import PdfReader

import re

@traceable(name="retrieve_document")
def pdf_to_sentences(pdf_path):
    reader=PdfReader(pdf_path)
    text=""

    for page in reader.pages:
        text+=page.extract_text()+" "

    text=text.replace("\n"," ")

    sentences=re.split(r'(?<=[.!?])\s+',text)

    sentences=[s.strip() for s in sentences if s.strip()]

    return sentences

SOURCE_PDF="/home/abrar/Desktop/Abrar/LangChain/Documents/the-theories-and-fatality-of-bermuda-triangle-52775.pdf"

document=pdf_to_sentences(SOURCE_PDF)
print(len(document))

#print(document[:5])

embedding=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

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
model=ChatHuggingFace(llm=llm)

def generate_embeddings(document):
    if isinstance(document,list):
        return embedding.embed_documents(document)
    else:
        return embedding.embed_query(document)


def calling_llm(question,answer):
    prompt=f'''
you are an (Retrieval Augmented Generation) RAG assistant 
RULES:
- Answer ONLY using the provided context.
- if you don't find context say i don't know
- Do NOT use external knowledge.
- Be concise and factual.
context:{answer}
question:{question}
Give Answer
'''
    result=model.invoke(prompt)
    return result.content


doc_emb=generate_embeddings(document)



question = 'what was on board  USS Cyclops (1918) a 542-foot-long Navy cargoship ,that  sank somewhere between Barbados and the Chesapeake Bay'


ques_emb=generate_embeddings(question)

scores=cosine_similarity([ques_emb],doc_emb)[0]
top_k=scores.argsort()[::-1][:12]

"""
scores.agrsort() sorts indices in asscecding order , [::-1] make them in descending order, [:12] returns top 12 indices
"""

from langsmith import traceable
@traceable(name="load_and_retrieve")
def retrieve_context(question, k=12):
    ques_emb = embedding.embed_query(question)
    scores = cosine_similarity([ques_emb], doc_emb)[0]
    top_k = scores.argsort()[::-1][:k]
    return "\n".join([document[i] for i in top_k])


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a Retrieval Augmented Generation (RAG) assistant.

Rules:
- Answer ONLY using the provided context
- If the answer is not in the context, say "I don't know"
- Do NOT use external knowledge
- Be concise and factual

Context:
{context}

Question:
{question}

Answer:
""")
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {
        "context": lambda x: retrieve_context(x["question"]),
        "question": RunnablePassthrough()
    }
    | prompt
    | model
)


Answers=[]

for i in top_k:
    Answers+=[document[i]]
#print(Answers)


print("===LLM ANSWER===")

result=rag_chain.invoke({
    "question":question
})
print(result.content)
