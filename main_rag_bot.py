import os
import io
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_groq import ChatGroq

# 🧠 Set Tesseract path if on Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 🔐 Load environment variables
load_dotenv()
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION_NAME")


# 📄 Extract text from normal + scanned PDFs
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if text.strip():  # If text extraction worked
            return text
    except:
        pass  # Fallback to OCR

    # OCR fallback using PyMuPDF
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        text += pytesseract.image_to_string(img)
    return text


# ✂️ Split text into chunks
def split_text(text: str, chunk_size=1000, chunk_overlap=200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])


# 🧠 Store in Qdrant Cloud
def store_in_qdrant(documents: List[Document]):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Qdrant.from_documents(
        documents,
        embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name
    )
    return vectorstore


# 🤖 RAG QA pipeline setup
def setup_retrieval_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=groq_api_key
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa


# 🚀 Process folder of PDFs
def process_folder(folder_path: str) -> List[Document]:
    all_documents = []
    for file_path in Path(folder_path).rglob("*.pdf"):
        print(f"Processing: {file_path}")
        text = extract_text_from_pdf(str(file_path))
        documents = split_text(text)
        for doc in documents:
            doc.metadata["source"] = str(file_path)  # Add source metadata
        all_documents.extend(documents)
    return all_documents


# 🧠 Interactive Chat with Docs
def chat_with_docs(qa_chain):
    print("\n🤖 Start chatting with your documents! Type 'exit' to quit.")
    while True:
        query = input("🧑‍💻 You: ")
        if query.lower() in ("exit", "quit"):
            print("👋 Exiting chat.")
            break
        result = qa_chain.invoke(query)
        print("🤖 AI:", result["result"])
        print("📚 Sources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source", "Unknown"))


# -------------------- MAIN CONFIG --------------------
if __name__ == "__main__":
    folder_path = r"C:\Users\ACER\OneDrive\Desktop\AI_Clone\docs"  # You can move this to .env too if needed

    # Step 1: Ingest
    documents = process_folder(folder_path)

    # Step 2: Embed & Store
    vectorstore = store_in_qdrant(documents)

    # Step 3: Setup QA
    qa_chain = setup_retrieval_chain(vectorstore)

    # Step 4: Chat
    chat_with_docs(qa_chain)
