import os
import requests
import json
import uuid
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant  # ✅ Fixed Import
from langchain_community.embeddings import HuggingFaceEmbeddings  # ✅ Fixed Import
from langchain.llms.base import LLM
from typing import Any, List, Optional

# Load environment variables
load_dotenv()

# Load API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ARIZE_API_KEY = os.getenv("ARIZE_API_KEY")
ARIZE_SPACE_ID = os.getenv("ARIZE_SPACE_ID")

print("✅ API Keys Loaded!")

# ------------------------------
# 1️⃣ Set up Qdrant (Vector Database)
# ------------------------------
collection_name = "ai_clone"
client = QdrantClient(path="qdrant_db")  # ✅ Uses persistent storage

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

print(f"✅ Qdrant collection '{collection_name}' is ready!")

# ------------------------------
# 2️⃣ Load Embeddings & Vector Store
# ------------------------------
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
lc_vectorstore = Qdrant(client, collection_name, hf_embeddings)

# ------------------------------
# 3️⃣ Custom LLM Wrapper for Groq API
# ------------------------------
class GroqLLM(LLM):
    api_key: str
    endpoint: str = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(self.endpoint, headers=headers, json=data)
            response.raise_for_status()
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "Error in response")
        except requests.exceptions.RequestException as e:
            return f"❌ Groq API error: {e}"

# ✅ Initialize the LLM with API key
groq_llm = GroqLLM(api_key=GROQ_API_KEY)

# ------------------------------
# 4️⃣ Build Retrieval QA Chain
# ------------------------------
retriever = lc_vectorstore.as_retriever(search_kwargs={"k": 10, "score_threshold": 0.2})  # Lower threshold

qa_chain = RetrievalQA.from_chain_type(
    llm=groq_llm,
    chain_type="stuff",
    retriever=retriever
)

def get_answer(question: str) -> str:
    """Retrieve relevant AI knowledge and generate an answer."""

    print(f"\n🔍 Searching Qdrant for: {question}")

    # 🔍 Step 1: Try to Retrieve from Qdrant
    docs = retriever.invoke(question)

    if not docs:
        return "❌ No relevant data found in Qdrant!"

    # 🔍 Step 2: Print Retrieved Documents
    print("\n✅ Retrieved Documents from Qdrant:")
    for i, doc in enumerate(docs[:3]):  # Show first 3 results
        print(f"{i+1}. {doc.page_content[:300]}...\n")

    # 🔍 Step 3: Generate Answer Using AI
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Using the retrieved AI knowledge:\n{context}\nQuestion: {question}\nAnswer:"
    
    response = qa_chain.invoke(prompt)

    # 🔍 Fix output format
    if isinstance(response, dict) and "result" in response:
        return f"**AI Answer:** {response['result']}\n\n🔹 _Retrieved from knowledge base._"

    return response  # If response is already a string

# ------------------------------
# 5️⃣ Arize AI Logging (Fixed)
# ------------------------------
try:
    from arize.pandas.logger import Client, Schema
    from arize.utils.types import ModelTypes, Environments

    ARIZE_ENVIRONMENT = Environments.PRODUCTION
    ARIZE_MODEL_ID = os.getenv("ARIZE_MODEL_ID", "AI_Clone_Chatbot")
    ARIZE_MODEL_TYPE = ModelTypes.GENERATIVE_LLM

    arize_schema = Schema(
        prediction_id_column_name="prediction_id",
        timestamp_column_name="prediction_ts",
        prediction_label_column_name="prediction_label",
        actual_label_column_name="actual_label",  # ✅ Fixed Missing Field
        feature_column_names=["question"]
    )

    arize_client = Client(space_id=ARIZE_SPACE_ID, api_key=ARIZE_API_KEY)

    def log_to_arize(question: str, answer: str):
        """Log chatbot interactions to Arize AI with an actual label for evaluation."""
        
        # 🔹 Define the expected correct answer if available (for evaluation)
        actual_answers = {
            "What is Retrieval-Augmented Generation?": 
            "Retrieval-Augmented Generation (RAG) is a technique that enhances AI responses by retrieving external knowledge before generating an answer."
        }

        # 🔹 Assign actual label if available; otherwise, set to "UNKNOWN"
        actual_label = actual_answers.get(question, "UNKNOWN")
            
        df = pd.DataFrame({
            "prediction_id": [str(uuid.uuid4())],  # Unique ID for tracking
            "prediction_ts": [datetime.now(timezone.utc)],
            "question": [question],
            "prediction_label": [answer],  # AI-generated response
            "actual_label": [actual_label]  # Correct answer (if available)
        })      

        try:
            arize_client.log(
                dataframe=df,
                schema=arize_schema,
                model_id=ARIZE_MODEL_ID,
                model_version="v1",
                model_type=ARIZE_MODEL_TYPE,
                environment=ARIZE_ENVIRONMENT
            )
            print("✅ Logged evaluation to Arize AI!")
        except Exception as e:
            print(f"❌ Error logging to Arize: {e}")

except ImportError:
    def log_to_arize(question: str, answer: str):
        print("⚠️ Arize AI not installed or configured.")

# ------------------------------
# 6️⃣ Run Example Chatbot Interaction
# ------------------------------
if __name__ == "__main__":
    user_question = "What is Retrieval-Augmented Generation?"
    answer = get_answer(user_question)
    
    print("\n🤖 AI Clone Chatbot")
    print(f"Q: {user_question}")
    print(f"A: {answer}")

    log_to_arize(user_question, answer)
