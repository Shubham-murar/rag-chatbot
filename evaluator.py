# evaluate_bot.py

import os
from dotenv import load_dotenv
import re
import json
from main_rag_bot import setup_retrieval_chain
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

# Load environment variables from .env file
load_dotenv()

# Fetch credentials from environment
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION_NAME")

# Normalize text for more reliable keyword matching
def normalize(text):
    return re.sub(r'[^\w\s]', '', text.lower())

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
        "expected_keywords": ["retrievalaugmented", "generation", "knowledgeintensive", "nlp"]
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

# Setup vectorstore and QA chain
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
vectorstore = Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)
qa_chain = setup_retrieval_chain(vectorstore, groq_api_key)

# Run evaluation
total_score = 0
results = []

print("🧪 Starting Evaluation...\n")

for case in test_cases:
    question = case["question"]
    expected_keywords = [normalize(k) for k in case["expected_keywords"]]

    try:
        response = qa_chain.invoke(question)["result"]
    except Exception as e:
        response = f"Error: {str(e)}"

    if not response:
        response = "No response generated."

    response_clean = normalize(response)
    matched_keywords = sum(1 for kw in expected_keywords if kw in response_clean)
    score = matched_keywords / len(expected_keywords) * 5
    total_score += score

    result = {
        "question": question,
        "response": response,
        "matched_keywords": matched_keywords,
        "total_keywords": len(expected_keywords),
        "score": round(score, 2)
    }
    results.append(result)

    # Print per case
    print(f"🔹 Q: {question}")
    print(f"🔸 A: {response}")
    print(f"✅ Matched Keywords: {matched_keywords} / {len(expected_keywords)} → Score: {score:.1f}/5\n")

# Final Score
average_score = total_score / len(test_cases)
print(f"🎯 Final Score: {average_score:.2f} / 5.0")

# Export results to JSON file
output_data = {
    "final_score": round(average_score, 2),
    "evaluations": results
}

output_file = "evaluation_results.json"
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"📄 Evaluation results saved to {output_file}")
