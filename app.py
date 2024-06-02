import json
import os 
import sys
import streamlit as st


from QASystem.ingestion import huggingface_embeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain.vectorstores import FAISS

from QASystem.ingestion import data_ingestion,get_vector_store

from QASystem.retrievalandgeneration import get_response_llm
from langchain_community.llms import CTransformers



def main():
    st.set_page_config("QA with Doc")
    st.header("QA with Doc using langchain and llama2")
    
    user_question=st.text_input("Ask a question from the pdf files")
    
    with st.sidebar:
        st.title("update or create the vector store")
        if st.button("vectors update"):
            with st.spinner("processing..."):
                docs=data_ingestion()
                get_vector_store(docs)
                st.success("done")
                
        if st.button("llama model"):
            with st.spinner("processing..."):
                faiss_index=FAISS.load_local("faiss_index",huggingface_embeddings,allow_dangerous_deserialization=True)
                llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q2_K.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.01})
                
                st.write(get_response_llm(llm,faiss_index,user_question))
                st.success("Done")
                
if __name__=="__main__":
    #this is my main method
    main()
    