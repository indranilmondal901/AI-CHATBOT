
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------------------------
# 1. Embedding model
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
    persist_directory="./chroma_db",
)


# --------------------------------------------------
# 3. Text splitter
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


# --------------------------------------------------
# 4. Ingest uploaded document
# --------------------------------------------------

def ingest_document(
    document_id: str,
    file_name: str,
    text: str,
    progress_callback=None,
):
    if not text.strip():
        raise ValueError("The document is empty.")

    # Create chunks
    chunks = text_splitter.split_text(text)

    if not chunks:
        raise ValueError("No text chunks were created.")

    if progress_callback:
        progress_callback("Chunking done")

    # Create LangChain Document objects
    documents = []

    for i, chunk in enumerate(chunks):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "document_id": document_id,
                    "source": file_name,
                    "chunk_id": i,
                },
            )
        )

    # Generate embeddings and store documents.
    # Chroma calls the embedding model and persists the records.
    vector_store.add_documents(documents)

    if progress_callback:
        progress_callback("Vectorization and ChromaDB storage done")

    return {
        "document_id": document_id,
        "file_name": file_name,
        "chunks": len(chunks),
        "status": "Ready for chat",
    }