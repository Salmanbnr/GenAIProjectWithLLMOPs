from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock

import json
import os
import sys
import boto3

bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)

def data_ingestion():
    loader = PyPDFDirectoryLoader("./Data")
    document = loader.load()
    # Optimized text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(document)
    return docs

def get_vector_store(docs, rebuild=False):
    if os.path.exists("faiss_index") and not rebuild:
        return FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
    else:
        vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)
        vector_store_faiss.save_local("faiss_index")
        return vector_store_faiss

if __name__ == '__main__':
    docs = data_ingestion()
    get_vector_store(docs)