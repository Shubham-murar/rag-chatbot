

```markdown
# 🧠 Retrieval-Augmented Generation (RAG) Chatbot with Qdrant & Groq LLM

This project demonstrates a powerful RAG chatbot combining advanced document retrieval with generative language models. It uses Qdrant for vector storage, SentenceTransformers for embeddings, and Groq's `llama-3.1-8b-instant` model for response generation. The project includes semantic evaluation to assess answer quality.

---

## 🚀 Project Overview

RAG enhances language model responses by retrieving relevant knowledge chunks from a vector database and conditioning generation on that information. This implementation shows how to:

- Chunk documents for efficient retrieval.
- Embed text using `all-MiniLM-L6-v2` SentenceTransformer.
- Store and search embeddings in Qdrant vector database.
- Generate context-aware answers via Groq's LLM API.
- Evaluate answers semantically using SentenceTransformer embeddings and cosine similarity.

---

## 📂 Repository Structure

```

.
├── main\_rag\_bot.py              # Core RAG chatbot pipeline
├── evaluator.py                 # Semantic evaluation script
├── documents/                  # Source knowledge files (txt, pdf)
├── .env                        # Environment variables
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── evaluation\_results\_semantic.json # Semantic evaluation output

````

---

## 🧱 Core Technologies & Configurations

| Component            | Description                                         |
|----------------------|-----------------------------------------------------|
| **Chunking Method**  | Fixed-length overlapping chunks for context window  |
| **Embedding Model**  | `sentence-transformers/all-MiniLM-L6-v2`             |
| **Vector Database**  | Qdrant Cloud                                        |
| **LLM Model**        | Groq `llama-3.1-8b-instant`                         |
| **Evaluation**       | Semantic similarity scoring using SentenceTransformer embeddings and cosine similarity |

---

## ⚙️ Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Shubham-murar/rag-chatbot.git
   cd rag-chatbot
````

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**

   Create a `.env` file in the root directory with the following keys:

   ```
   QDRANT_URL=https://your-qdrant-url.com
   QDRANT_API_KEY=your-qdrant-api-key
   QDRANT_COLLECTION_NAME=your-collection-name
   GROQ_API_KEY=your-groq-api-key
   ```

4. **Add Your Documents**

   Place your `.txt` or `.pdf` knowledge files inside the `documents/` folder. These will be chunked, embedded, and uploaded to Qdrant.

5. **Run the Chatbot**

   Launch the RAG chatbot pipeline:

   ```bash
   python main_rag_bot.py
   ```

---

## 🧪 Semantic Evaluation

Evaluate the chatbot’s answer quality using semantic similarity against reference summaries:

```bash
python evaluator.py
```

This script computes cosine similarity between embeddings of chatbot answers and reference summaries, providing:

* Semantic scores scaled to 0-5 per question
* Average semantic score across all questions
* Saves detailed results in `evaluation_results_semantic.json`

Example output:

```
🎯 Final Semantic Score: 4.65 / 5.0
📄 Evaluation results saved to evaluation_results_semantic.json
```

---

## 📹 Project Demo Video

🎬 **Watch Demo:** [YouTube Demo Video](https://youtu.be/rhpk6ASrcmc)

Demo includes:

* Document chunking and embedding
* Qdrant vector search
* Chatbot interaction with Groq LLM
* Semantic evaluation and scoring
* Optional Arize AI integration for advanced analysis

---

## ✅ Contribution Guidelines

* Ensure all code is original and clearly documented.
* Explain your design choices, especially around chunking and evaluation.
* Include relevant test cases with pull requests.

---

## 🙋‍♂️ Author

**Shubham Ashok Murar**
📧 [shubhammurar3322@gmail.com](mailto:shubhammurar3322@gmail.com)
🌐 LinkedIn - www.linkedin.com/in/shubham-murar/ |
🐈‍⬛ [GitHub](https://github.com/Shubham-murar/rag-chatbot)

