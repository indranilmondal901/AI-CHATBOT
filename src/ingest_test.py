'''
STEP 1: Read document  
STEP 2: Create Text Splitter
STEP 3: Split the document
'''

# Chunking for llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# STEP 1: Read document
with open("../data/services.txt", "r", encoding="utf-8") as file:
    text = file.read();

# STEP 2: Create Text Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50);

# STEP 3: Split the document
chunks = text_splitter.split_text(text);

'''
# Print chunks
# print(f"Total chunks: {len(chunks)}")
# for i, chunk in enumerate(chunks):
#     print("\n" + "=" * 60)
#     print(f"CHUNK {i + 1}---> {len(chunk)}") # recursive
#     print("=" * 60)
#     print(chunk)
#     print(len(chunk));
'''

# STEP 4: LOAD EMBEDDING MODEL
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# STEP 5: GENERATE EMBEDDINGS
embeddings  = embedding_model.encode(chunks);

# STEP 6: INSPECT RESULTS
for i, (chunk, embedding) in enumerate(
    zip(chunks, embeddings)
):

    print("\n" + "=" * 60)
    print(f"CHUNK {i + 1}")
    print("=" * 60)

    print("Text:")
    print(chunk)

    print("\nEmbedding dimensions:")
    print(len(embedding))

    print("\nFirst 5 numbers:")
    print(embedding)
