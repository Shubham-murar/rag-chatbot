import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Fixed Import
from langchain_community.document_loaders import TextLoader  # ✅ Fixed Import
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load extracted knowledge base
TEXT_FILE_PATH = r"C:\Users\ACER\OneDrive\Desktop\AI_Clone\docs\knowledge_base.txt"

# Initialize Qdrant Client
client = QdrantClient(path="qdrant_db")  # ✅ FIXED: Uses persistent storage


# Collection name
collection_name = "ai_clone"

# Create collection if not exists
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

# Load text from file
loader = TextLoader(TEXT_FILE_PATH, encoding="utf-8")
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Initialize Embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store chunks in Qdrant
for i, chunk in enumerate(chunks):
    vector = hf_embeddings.embed_query(chunk.page_content)
    client.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=i, vector=vector, payload={"page_content": chunk.page_content})]
    )

print(f"✅ {len(chunks)} AI knowledge chunks stored in Qdrant!")

# 🔹 Debugging: Check if data was stored
# 🔍 Check if data exists in Qdrant before running chatbot
retrieved_points = client.scroll(collection_name=collection_name, limit=3)[0]

if retrieved_points:
    print("\n✅ Data Exists in Qdrant! Here are some stored chunks:")
    for point in retrieved_points:
        print(f"- {point.payload.get('page_content', 'No content found')[:200]}...\n")
else:
    print("\n❌ No data found in Qdrant! Retrieval will fail.")

