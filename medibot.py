import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL = "llama-3.3-70b-versatile"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know — don't try to make up an answer.
Do not provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(api_key: str):
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.5, api_key=api_key)
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
    retriever = get_vectorstore().as_retriever(search_kwargs={"k": 3})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def main():
    st.title("Ask Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")
    if not prompt:
        return

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY is not set. Get one at https://console.groq.com/keys and add it to your .env file.")
        return

    try:
        chain, _ = build_chain(groq_api_key)
        result = chain.invoke(prompt)
        st.chat_message("assistant").markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})
    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
