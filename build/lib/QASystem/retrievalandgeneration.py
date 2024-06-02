
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from QASystem.ingestion import get_vector_store
from langchain_community.llms import CTransformers
from ingestion import huggingface_embeddings


prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but usse atleast summarize with 
250 words with detailed explainations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q2_K.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.01})


def get_response_llm(llm,vectorstore_faiss,query):
    qa=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
        
        
    )
    answer=qa({"query":query})
    return answer["result"]
    
if __name__=='__main__':
    # vectorstore_faiss= "./faiss_index.faiss"
    faiss_index=FAISS.load_local("faiss_index",huggingface_embeddings,allow_dangerous_deserialization=True)
    query="what is RAG token?"
    llm=llm
    get_response_llm(llm,faiss_index,query)
    


