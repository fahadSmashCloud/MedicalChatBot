from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src import pinecone_store

load_dotenv(find_dotenv())

DATA_PATH = "Data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


def load_pdf_files(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


if __name__ == "__main__":
    documents = load_pdf_files(data=DATA_PATH)
    print(f"Loaded documents: {len(documents)}")

    text_chunks = create_chunks(documents)
    print(f"Chunks: {len(text_chunks)}")

    embedding_model = get_embedding_model()

    if pinecone_store.is_configured():
        store = pinecone_store.PineconeVectorStore(embedding_model)
        n = store.add_documents(text_chunks)
        print(f"Upserted {n} chunks to Pinecone index '{pinecone_store.index_name()}'")
    else:
        db = FAISS.from_documents(text_chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        print(f"FAISS index saved to {DB_FAISS_PATH}")
