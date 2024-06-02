from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


import json
import os
import sys


# Initialize Hugging Face embeddings
huggingface_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def data_ingestion():
    loader = PyPDFDirectoryLoader("./data")
    documents = loader.load()
    
    if not documents:
        print("No documents loaded.")
        return []

    print(f"Loaded {len(documents)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    if not docs:
        print("No documents after splitting.")
        return []

    print(f"Split into {len(docs)} documents.")
    
    return docs

def get_vector_store(docs):
    # Extract text content from documents
    texts = [doc.page_content for doc in docs]

    # Create vector store from texts
    vector_store_faiss = FAISS.from_texts(texts, huggingface_embeddings)
    if not vector_store_faiss:
        print("Failed to create vector store.")
        return None

    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss

if __name__ == '__main__':
    docs = data_ingestion()
    if docs:  # Check if docs are loaded and split correctly
        get_vector_store(docs)
    else:
        print("No documents were loaded or split.")




# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain_community.embeddings import BedrockEmbeddings
# from langchain.llms.bedrock import Bedrock

# import json
# import os
# import sys
# import boto3## bedrock client

# bedrock=boto3.client(service_name="bedrock-runtime")
# bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


# def data_ingestion():
#     loader=PyPDFDirectoryLoader("./data")
#     documents=loader.load()
    
    
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
#     text_splitter.split_documents(documents)
    
#     docs=text_splitter.split_documents(documents)
    
#     return docs


# def get_vector_store(docs):
#     vector_store_faiss=FAISS.from_documents(docs,bedrock_embeddings)
#     vector_store_faiss.save_local("faiss_index")
#     return vector_store_faiss
    
# if __name__ == '__main__':
#     docs=data_ingestion()
#     get_vector_store(docs)
    
    