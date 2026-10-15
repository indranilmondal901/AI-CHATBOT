# Budhana AI — RAG Knowledge Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **FastAPI, LangChain, ChromaDB, Hugging Face embeddings, Ollama, and React**.

The application allows users to upload documents, process them into searchable vector embeddings, and ask questions based on the uploaded knowledge base. It also includes a LangSmith monitoring dashboard for inspecting LLM traces and application activity.

## 1. Project Overview

The primary objective of this project is to build a chatbot that answers questions using information from uploaded documents rather than relying only on the LLM's pretrained knowledge.

The chatbot retrieves relevant document chunks from ChromaDB and provides them as context to the language model before generating an answer.

### Key Features

* Upload TXT and PDF documents.
* Extract text from uploaded documents.
* Split document text into smaller chunks.
* Generate embeddings using a Hugging Face sentence-transformer model.
* Store and retrieve document chunks using ChromaDB.
* Ask questions through a React-based chat interface.
* Generate context-based responses using a locally running Ollama model.
* Display document sources associated with chatbot responses.
* View uploaded documents and their processing statuses.
* Delete documents and their associated vector records.
* Monitor application traces through a LangSmith dashboard.
* View trace summaries, errors, latency, token usage, run details, and nested child runs where available.

## 2. Technology Stack

| Component           | Technology                                    |
| ------------------- | --------------------------------------------- |
| Frontend            | React, Vite, JavaScript                       |
| UI Icons            | Lucide React                                  |
| Backend             | Python, FastAPI                               |
| LLM Integration     | LangChain                                     |
| Local LLM           | Ollama — `llama3.2:3b`                        |
| Embedding Model     | `sentence-transformers/all-MiniLM-L6-v2`      |
| Vector Database     | ChromaDB                                      |
| Document Processing | LangChain text splitters, PDF text extraction |
| LLM Observability   | LangSmith                                     |
| Configuration       | Environment variables using `.env`            |

## 3. Project Structure

```text
budhana-rag-chatbot/
│
├── data/
│   ├── services.txt
│   ├── technologies.txt
│   ├── staffing.txt
│   ├── policies.txt
│   ├── uploads/
│   └── documents.json
│
├── src/
│   ├── ingest.py
│   ├── retrieve.py
│   ├── rag.py
│   └── api.py
│
├── chroma_db/
│
├── evaluation/
│   ├── test_dataset.json
│   └── run_evaluation.py
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── TopBar.jsx
│   │   │   ├── DocumentPanel.jsx
│   │   │   ├── UploadDocument.jsx
│   │   │   ├── ChatPanel.jsx
│   │   │   ├── MessageBubble.jsx
│   │   │   ├── LangSmithDashboard.jsx
│   │   │   └── LangSmithDashboard.css
│   │   │
│   │   ├── services/
│   │   │   └── api.js
│   │   │
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   │
│   └── package.json
│
├── .env
├── requirements.txt
└── README.md
```

## 4. How the Application Works

### Document Ingestion Pipeline

1. The user uploads a TXT or PDF document through the frontend.
2. FastAPI receives the uploaded file.
3. The backend extracts the document text.
4. `RecursiveCharacterTextSplitter` divides the text into chunks.
5. Hugging Face generates an embedding vector for each chunk.
6. The chunks, embeddings, and metadata are stored in ChromaDB.
7. The document registry tracks uploaded documents and their processing status.

The current text-splitting configuration is:

* **Chunk size:** 500 characters
* **Chunk overlap:** 50 characters

### Question Answering Pipeline

1. The user submits a question in the chat interface.
2. The backend converts the question into an embedding.
3. ChromaDB searches for semantically relevant document chunks.
4. The retrieved context is supplied to the LLM.
5. Ollama generates a response using the retrieved information.
6. The frontend displays the answer and available source information.

The chatbot should indicate when the knowledge base does not contain sufficient information rather than inventing unsupported facts.

### LangSmith Monitoring

LangSmith is configured to trace supported LangChain and LLM operations.

The monitoring dashboard currently provides:

* Run and trace summaries.
* Error and latency metrics.
* Token usage information when available.
* A list of recent root runs.
* Run status filtering and pagination.
* Individual run details, including available inputs, outputs, metadata, and child runs.

Metrics depend on the data available from LangSmith and the instrumentation implemented in the application.

## 5. Prerequisites

Install the following before running the project:

* Python 3.10 or a compatible version for the installed dependencies.
* Node.js and npm.
* Ollama.
* Git (optional).

Check the installed versions:

```bash
python --version
node --version
npm --version
ollama --version
```

## 6. Installation and Setup

### Step 1: Clone or open the project

Open a terminal in the project root directory.

```bash
cd budhana-rag-chatbot
```

### Step 2: Create a Python virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the backend dependencies:

```bash
pip install -r requirements.txt
```

### Step 3: Configure Ollama

Install Ollama if it is not already installed, then download the model:

```bash
ollama pull llama3.2:3b
```

Ensure Ollama is running and the model is available:

```bash
ollama list
```

### Step 4: Configure environment variables

Create or update the `.env` file in the project root.

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=budhana-rag-chatbot
```

Replace the placeholder with your own LangSmith API key.

**Security:** Never commit `.env` or expose your LangSmith API key in the React frontend. Add `.env`, `.venv/`, and other local artifacts to `.gitignore`.

LangSmith tracing can be disabled when required by setting `LANGSMITH_TRACING=false`.

### Step 5: Start the backend

From the project root:

```bash
python -m uvicorn src.api:app --reload
```

The backend should be available at:

* API base URL: `http://127.0.0.1:8000`
* Swagger documentation: `http://127.0.0.1:8000/docs`

### Step 6: Install frontend dependencies

Open a second terminal:

```bash
cd frontend
npm install
```

### Step 7: Start the frontend

```bash
npm run dev
```

Open the local URL displayed by Vite, commonly:

`http://localhost:5173`

Keep the backend and frontend terminals running while using the application.

## 7. API Endpoints

The backend includes endpoints for health checks, document management, chat, and LangSmith monitoring.

| Method | Endpoint                                  | Purpose                                  |
| ------ | ----------------------------------------- | ---------------------------------------- |
| GET    | `/docs`                                   | Open interactive API documentation       |
| GET    | `/api/health`                             | Check backend health                     |
| GET    | `/api/documents`                          | Retrieve uploaded document records       |
| GET    | `/api/documents/{document_id}/status`     | Check document processing status         |
| POST   | `/api/documents/upload`                   | Upload a document                        |
| DELETE | `/api/documents/{document_id}`            | Delete a document and its vector records |
| POST   | `/api/chat`                               | Submit a chatbot question                |
| GET    | `/api/monitoring/langsmith`               | Check LangSmith monitoring status        |
| GET    | `/api/monitoring/langsmith/summary`       | Retrieve trace summary metrics           |
| GET    | `/api/monitoring/langsmith/runs`          | Retrieve and filter runs                 |
| GET    | `/api/monitoring/langsmith/runs/{run_id}` | Retrieve individual run details          |

Refer to `/docs` for the exact request parameters and payload formats supported by the running backend.

## 8. Configuration Notes

* The frontend API helper currently uses `http://127.0.0.1:8000` as its backend base URL.
* FastAPI CORS settings must allow the origin used by the Vite development server.
* ChromaDB uses persistent local storage under `chroma_db/`.
* Uploaded documents are stored under `data/uploads/`.
* Document metadata is maintained in `data/documents.json`.
* LangSmith monitoring requires valid backend configuration and accessible LangSmith services.
* Token, cost, and latency metrics may be unavailable for some runs, depending on what the provider reports.

## 9. Current Development Status

### Implemented

* [x] FastAPI backend setup.
* [x] React and Vite frontend.
* [x] TXT and PDF document upload.
* [x] Document ingestion and chunking pipeline.
* [x] Hugging Face embedding integration.
* [x] Persistent ChromaDB vector storage.
* [x] Retrieval and RAG-based chat integration.
* [x] Ollama local LLM integration.
* [x] Document status and deletion functionality.
* [x] LangSmith configuration and monitoring API endpoints.
* [x] Basic LangSmith monitoring dashboard.
* [x] Trace listing, filtering, pagination, and detail inspection.

### Future Improvements

* [ ] Add more advanced LangSmith dashboard charts and time-range filters.
* [ ] Improve retrieval evaluation and answer-quality measurements.
* [ ] Add more comprehensive automated tests.
* [ ] Improve authentication and production deployment security.
* [ ] Optimize retrieval quality and document-processing performance.

## 10. Troubleshooting

**Backend fails to start**

* Activate the Python virtual environment.
* Install dependencies from `requirements.txt`.
* Check the terminal for import or configuration errors.

**Ollama connection or model error**

* Confirm Ollama is running.
* Verify that `llama3.2:3b` is available using `ollama list`.

**Frontend cannot reach the backend**

* Confirm the FastAPI server is running on port `8000`.
* Verify the base URL in `frontend/src/services/api.js`.
* Check CORS configuration and browser console errors.

**Document ingestion fails**

* Check the backend terminal for the original exception.
* Confirm the uploaded file is supported and contains extractable text.
* Verify that the backend can write to the upload and ChromaDB directories.

**LangSmith dashboard shows no data**

* Verify the API key and project name in `.env`.
* Confirm tracing is enabled and the backend has been restarted after environment changes.
* Generate some chatbot requests and check whether runs appear in the configured LangSmith project.
* Remember that manually executed vector database operations may require explicit tracing instrumentation.

## 11. Project Goal

The goal of Budhana AI is to provide a document-grounded knowledge assistant with local LLM inference, semantic search, document management, and observability through LangSmith.

This project serves as a practical implementation of RAG concepts, vector databases, embedding models, API development, frontend integration, and LLM monitoring.
