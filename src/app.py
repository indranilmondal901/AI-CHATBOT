# with open("../data/services.txt", "r", encoding="utf-8") as file:
#     text = file.read()

# # print(text);

# Chunking for llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# STEP 1: Read document
# --------------------------------------------------

with open("../data/services.txt", "r", encoding="utf-8") as file:
    text = file.read()

# print(text);

# --------------------------------------------------
# STEP 2: Create Text Splitter
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# --------------------------------------------------
# STEP 3: Split the document
# --------------------------------------------------

chunks = text_splitter.split_text(text)

print(type(chunks))
# <class 'list'>

# --------------------------------------------------
# STEP 4: Print chunks
# --------------------------------------------------

print(f"Total chunks: {len(chunks)}")

# for i, chunk in enumerate(chunks):
#     print("\n" + "=" * 60)
#     print(f"CHUNK {i + 1}")
#     print("=" * 60)
#     print(chunk)
