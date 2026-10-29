import streamlit as st
#import os
#from dotenv import load_dotenv
#LangChain Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#-----------------------
# page config
#-----------------------
st.set_page_config(page_title="C++ RAG Chatbot")
st.title("C++ RAG Chatbot")
st.write("Ask any questions related to C++ introduction.")

#-----------------------
#Cache Document Loading
#----------------------
@st.cache_resource
def load_vectorstore():
    #Load Document
    loader=TextLoader("CPP.txt",encoding="utf-8")
    documents= loader.load()

    #Split Text
    textSplitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
        )
    finalDocuments=textSplitter.split_documents(documents)
    finalDocuments

    #Embeddings
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print(embeddings)

    db= FAISS.from_documents(finalDocuments, embeddings)
    return db

#load vector DB (only once)
db = load_vectorstore()

#------------------
#user input and query
#------------------

query = st.text_input("enter your question about C++ introduction")

if query:
    docs = db.similarity_search(query, k=3)
    st.subheader("Retrieved Context")
    for i, doc in enumerate(docs):
        st.markdown(f"**Result {i+1}:**")
        st.write(doc.page_content)