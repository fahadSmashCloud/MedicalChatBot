from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


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

    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    print(f"FAISS index saved to {DB_FAISS_PATH}")
