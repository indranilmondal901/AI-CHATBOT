'''
Retrive relevent chunk from chromaDB
'''
from langchain_chroma import Chroma;
from langchain_huggingface import HuggingFaceEmbeddings;

# 1. Load the same embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to existing ChromaDB
vector_store = Chroma(
    collection_name="budhana_knowledgebase",
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)

# 3. User question
question = "What technologies does Budhana Tech use?"


# 4. Search ChromaDB
results = vector_store.similarity_search(
    question,
    k=3
);

# 5. Display results
print("\nRetrieved Documents:\n")

for i, doc in enumerate(results, start=1):

    print(f"--- Result {i} ---")

    print("Source:", doc.metadata.get("source"))
    print("Chunk ID:", doc.metadata.get("chunk_id"))

    print("Content:")
    print(doc.page_content)

    print()