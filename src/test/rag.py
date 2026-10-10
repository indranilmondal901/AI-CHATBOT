from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

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
    persist_directory="./chroma_db"
)

# --------------------------------------------------
# 3. Create Retriever
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)

# --------------------------------------------------
# 4. Load LLM
# --------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)

# --------------------------------------------------
# 5. Prompt Template
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a Budhana Tech knowledge assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer cannot be found in the context,
say that the information is not available in the
provided knowledge base.

Context:
{context}

Question:
{question}
""")
# --------------------------------------------------
# 6. Format retrieved documents
# --------------------------------------------------

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# --------------------------------------------------
# 7. Create RAG Chain
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        # "stream": lambda _: True,
    }
    | prompt
    | llm
)


# --------------------------------------------------
# 8. Ask a question
# --------------------------------------------------

if __name__ == "__main__":
    print("=== Budhana Tech Knowledge Chatbot ===")
    print("Ask a question, or type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break

        if not question:
            continue

        try:
            response = rag_chain.invoke(question)
            print("\nChatbot:", response.content, "\n")
        except Exception as error:
            print("\nAn error occurred:", error, "\n")