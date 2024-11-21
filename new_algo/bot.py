import os
import sys
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredFileLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from flask import Flask, request, jsonify
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
import shutil


loader = DirectoryLoader("../corpus_txt_agg/", loader_cls=UnstructuredFileLoader)
document = loader.load()

def embeddings(name):
   return HuggingFaceEmbeddings(
            model_name='name',
            model_kwargs={'device': 'cuda:0'}
        )
   
local_embeddings = OllamaEmbeddings(model="zylonai/multilingual-e5-large")

db_name = "/mnt/diskSustainability/GOM/RAG/db_e5"

model = ChatOllama(
            model="mixtral:8x7b",
        )

template = """
            dit moi quels jour somme nous
          """
prompt = ChatPromptTemplate.from_template(template)
question_answer_chain = create_stuff_documents_chain(model, prompt)
print("hello")