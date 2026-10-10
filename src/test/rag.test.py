from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# --------------------------------------------------
# 1. Load embedding model
# --------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 2. Connect to ChromaDB
# --------------------------------------------------

vector_store = Chroma(
    collection_name="budhana_knowledgebase",
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# 3. Load LLM
# --------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# --------------------------------------------------
# 4. User question
# --------------------------------------------------

question = input("type question:")


# --------------------------------------------------
# 5. Retrieve relevant documents
# --------------------------------------------------

results = vector_store.similarity_search(
    question,
    k=3
);
print(results);
# --------------------------------------------------
# 6. Create context
# --------------------------------------------------

context = "\n\n".join(
    doc.page_content
    for doc in results
)


# --------------------------------------------------
# 7. Create prompt
# --------------------------------------------------

prompt = f"""
You are a Budhana Tech assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer cannot be found in the context,
say that the information is not available in the
provided knowledge base.

Context:
{context}

Question:
{question}
"""
print(f"prompt: /n {prompt}")

# --------------------------------------------------
# 8. Generate answer
# --------------------------------------------------

response = llm.invoke(prompt)


# --------------------------------------------------
# 9. Display answer
# --------------------------------------------------

print("\nAnswer:")
print(response.content)