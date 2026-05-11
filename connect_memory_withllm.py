import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise SystemExit("GROQ_API_KEY is not set. Add it to your .env file.")

llm = ChatGroq(model=GROQ_MODEL, temperature=0.3, api_key=GROQ_API_KEY)
print("LLM Loaded")

CUSTOM_PROMPT_TEMPLATE = """
Use the provided context to answer the question.

If the answer is not in the context, say:
"I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
print("FAISS Loaded")

retriever = db.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

while True:
    query = input("\nQuery: ").strip()
    if query.lower() in ["exit", "quit"]:
        break
    print("\nANSWER:\n")
    print(chain.invoke(query))
