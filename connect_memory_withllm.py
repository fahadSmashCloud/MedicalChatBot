import os 
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Hugging Face API token
HF_TOKEN = os.environ.get("HG_TOKEN")
DB_FIAAS_PATH = "vectorstore/db_fiaas"

# Hugging Face Model ID
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# ✅ Fix: Correct way to initialize HuggingFaceEndpoint
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
         repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    temperature=0.5,
    model_kwargs={"max_length": 512},
    huggingfacehub_api_token=HF_TOKEN
    )
    return llm

# Load the LLM
llm = load_llm(HUGGINGFACE_REPO_ID)
print("✅ Hugging Face LLM loaded successfully!")


# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["question", "context"])

# Load FAISS embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

# ✅ Fix: Ensure FAISS DB exists before loading
if os.path.exists(DB_FIAAS_PATH):
    db = FAISS.load_local(DB_FIAAS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print("✅ FAISS database loaded successfully!")
else:
    print("⚠️ FAISS database not found at:", DB_FIAAS_PATH)
    exit(1)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  # ✅ No need to reload LLM again
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# ✅ Fix: Ensure correct input format
user_query = input("Write query here: ")

response = qa_chain.invoke(user_query)  # ✅ Fix: Pass only the query string

# ✅ Fix: Ensure response format
if isinstance(response, dict) and "result" in response:
    print("RESULT:", response["result"])
    print("SOURCE DOCUMENTS:", response.get("source_documents", []))
else:
    print("⚠️ Unexpected response format:", response)
