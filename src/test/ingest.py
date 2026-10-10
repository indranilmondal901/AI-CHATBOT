'''
Document load ---> Recursive text splitting ---> Creating chunk ---> embedding each chunk into vector ---> save into chroma db

'''
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# STEP 1: LOAD DOCUMENT
# ============================================================

with open("../data/services.txt", "r", encoding="utf-8") as file:
    text = file.read()


# ============================================================
# STEP 2: CREATE TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


# ============================================================
# STEP 3: CREATE CHUNKS
# ============================================================

chunks = text_splitter.split_text(text)

print(f"Total chunks: {len(chunks)}")


# ============================================================
# STEP 4: CONVERT CHUNKS INTO DOCUMENT OBJECTS
# ============================================================

documents = []

for i, chunk in enumerate(chunks):

    document = Document(
        page_content=chunk,
        metadata={
            "source": "services.txt",
            "chunk_id": i
        }
    )

    documents.append(document)


# ============================================================
# STEP 5: CREATE EMBEDDING MODEL
# ============================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# STEP 6: CREATE CHROMA VECTOR DATABASE
# ============================================================

vector_store = Chroma(
    collection_name="budhana_knowledgebase",
    embedding_function=embedding_model,
    persist_directory="./chroma_db"
)


# ============================================================
# STEP 7: ADD DOCUMENTS TO CHROMA
# ============================================================

vector_store.add_documents(documents);


print("\nDocuments successfully added to ChromaDB.")