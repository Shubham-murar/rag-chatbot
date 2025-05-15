import os
from dotenv import load_dotenv
import re
import json
from main_rag_bot import setup_retrieval_chain
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, util

# Load environment variables from .env file
load_dotenv()

# Fetch credentials from environment
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION_NAME")

# Normalize text function
def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

# Evaluation questions and expected keywords
test_cases = [
    {
        "question": "What is the main innovation introduced in BERT?",
        "expected_keywords": ["bidirectional", "pretraining", "transformers", "language understanding"]
    },
    {
        "question": "What are the key advantages of using GPUs in similarity search?",
        "expected_keywords": ["gpus", "similarity search", "scalability", "performance"]
    },
    {
        "question": "How does Dense Passage Retrieval improve open-domain question answering?",
        "expected_keywords": ["dense retrieval", "passages", "open-domain", "question answering"]
    },
    {
        "question": "What are the core features of LangChain?",
        "expected_keywords": ["langchain", "core features", "automation", "chain"]
    },
    {
        "question": "Explain how retrieval-augmented generation enhances knowledge-intensive NLP tasks.",
        "expected_keywords": ["retrieval-augmented", "generation", "knowledge-intensive", "nlp"]
    },
    {
        "question": "What does the research on scaling laws for neural language models focus on?",
        "expected_keywords": ["scaling laws", "neural language models", "scalability", "performance"]
    },
    {
        "question": "What are the advantages of using vector databases like Qdrant in RAG systems?",
        "expected_keywords": ["vector", "qdrant", "semantic search", "retrieval", "performance"]
    },
    {
        "question": "How does embedding size affect retrieval accuracy and speed?",
        "expected_keywords": ["embedding size", "accuracy", "retrieval", "speed", "tradeoff"]
    },
    {
        "question": "What role does chunking play in document retrieval for LLMs?",
        "expected_keywords": ["chunking", "retrieval", "context window", "llm", "granularity"]
    },
    {
        "question": "Explain the difference between dense and sparse retrieval techniques.",
        "expected_keywords": ["dense", "sparse", "retrieval", "vector", "bm25"]
    }
]

# Initialize embedding models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Setup Qdrant client and vectorstore
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
vectorstore = Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)
qa_chain = setup_retrieval_chain(vectorstore)

def semantic_score(answer, reference_text):
    """Calculate cosine similarity semantic score between answer and reference."""
    emb_answer = embedding_model.encode(normalize(answer), convert_to_tensor=True)
    emb_reference = embedding_model.encode(normalize(reference_text), convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(emb_answer, emb_reference)
    return float(cosine_sim)

# Run evaluation
total_score = 0
results = []

print("🧪 Starting Semantic Similarity Evaluation...\n")

for case in test_cases:
    question = case["question"]
    # Join keywords into a single reference summary sentence
    reference_text = " ".join(case["expected_keywords"])

    try:
        response = qa_chain.invoke(question)["result"]
    except Exception as e:
        response = f"Error: {str(e)}"

    if not response:
        response = "No response generated."

    score_raw = semantic_score(response, reference_text)  # cosine similarity 0-1
    score_scaled = round(score_raw * 5, 2)  # Scale 0 to 5

    total_score += score_scaled

    result = {
        "question": question,
        "response": response,
        "reference_summary": reference_text,
        "semantic_similarity": round(score_raw, 4),
        "score": score_scaled
    }
    results.append(result)

    print(f"🔹 Q: {question}")
    print(f"🔸 A: {response}")
    print(f"🔸 Reference Summary: {reference_text}")
    print(f"🔸 Semantic Similarity: {score_raw:.4f}")
    print(f"✅ Score: {score_scaled}/5\n")

# Final Average Score
average_score = total_score / len(test_cases)
print(f"🎯 Final Average Score: {average_score:.2f} / 5.0")

# Save results
output_data = {
    "final_score": round(average_score, 2),
    "evaluations": results
}

output_file = "evaluation_results_semantic.json"
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"📄 Evaluation results saved to {output_file}")
