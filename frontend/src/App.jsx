
import { useCallback, useEffect, useState } from "react";

import TopBar from "./components/TopBar";
import DocumentPanel from "./components/DocumentPanel";
import ChatPanel from "./components/ChatPanel";

import {
  checkHealth,
  getDocuments,
  getDocumentStatus,
  uploadDocument,
  deleteDocument,
  sendChatMessage,
} from "./services/api";
import LangSmithDashboard from "./components/LangSmithDashboard";

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState(null);
  const [apiOnline, setApiOnline] = useState(false);
  const [loadingDocuments, setLoadingDocuments] = useState(true);
  const [error, setError] = useState("");
  const [messages, setMessages] = useState([]);

  const [activeView, setActiveView] = useState("chat");

  // Load the document list only when requested.
  const refreshDocuments = useCallback(async (preferredId = null) => {
    const result = await getDocuments();
    const nextDocuments = result.documents || [];

    setDocuments(nextDocuments);

    setSelectedDocumentId((currentId) => {
      if (
        preferredId &&
        nextDocuments.some((doc) => doc.document_id === preferredId)
      ) {
        return preferredId;
      }

      if (
        currentId &&
        nextDocuments.some((doc) => doc.document_id === currentId)
      ) {
        return currentId;
      }

      return nextDocuments[0]?.document_id || null;
    });

    return nextDocuments;
  }, []);

  // Initial load: health check and document list, once per mount.
  useEffect(() => {
    let cancelled = false;

    async function initialize() {
      setLoadingDocuments(true);

      const healthPromise = checkHealth()
        .then((result) => {
          if (!cancelled) {
            setApiOnline(result.status === "ok");
          }
        })
        .catch(() => {
          if (!cancelled) setApiOnline(false);
        });

      const documentsPromise = refreshDocuments()
        .catch((err) => {
          if (!cancelled) {
            setError(err.message || "Could not load documents.");
          }
        })
        .finally(() => {
          if (!cancelled) setLoadingDocuments(false);
        });

      await Promise.all([healthPromise, documentsPromise]);
    }

    initialize();

    return () => {
      cancelled = true;
    };
  }, [refreshDocuments]);

  // Poll only documents currently marked as processing.
  const processingIds = documents
    .filter((doc) => doc.status === "processing")
    .map((doc) => doc.document_id)
    .sort()
    .join(",");

  useEffect(() => {
    if (!processingIds) return;

    let cancelled = false;
    let timeoutId;

    const ids = processingIds.split(",");

    async function pollStatuses() {
      const results = await Promise.allSettled(
        ids.map((id) => getDocumentStatus(id))
      );

      if (cancelled) return;

      const updatedDocuments = results
        .filter((result) => result.status === "fulfilled")
        .map((result) => result.value.document || result.value);

      if (updatedDocuments.length > 0) {
        setDocuments((previous) =>
          previous.map((doc) => {
            const updated = updatedDocuments.find(
              (item) => item.document_id === doc.document_id
            );

            return updated ? { ...doc, ...updated } : doc;
          })
        );
      }

      // Retry only while these documents remain in processing.
      // Updating a document to ready/failed changes processingIds,
      // causing this effect to clean up and stop polling.
      if (!cancelled) {
        timeoutId = setTimeout(pollStatuses, 2000);
      }
    }

    pollStatuses();

    return () => {
      cancelled = true;
      clearTimeout(timeoutId);
    };
  }, [processingIds]);

  async function handleUpload(file) {
    setError("");

    try {
      const result = await uploadDocument(file);

      // Refresh the list once after upload.
      // This also selects the newly uploaded document.
      await refreshDocuments(result.document_id);
    } catch (err) {
      setError(err.message || "Document upload failed.");
      throw err;
    }
  }

  async function handleDeleteDocument(documentId) {
    const document = documents.find(
      (item) => item.document_id === documentId
    );

    if (!document) return;

    const confirmed = window.confirm(
      `Delete "${document.filename}" and its indexed content?`
    );

    if (!confirmed) return;

    setError("");

    try {
      await deleteDocument(documentId);

      // One document-list request after successful deletion.
      await refreshDocuments();
    } catch (err) {
      setError(err.message || "Failed to delete document.");
    }
  }

  async function handleSendMessage(question) {
    if (!selectedDocumentId) return;

    setMessages((previous) => [
      ...previous,
      {
        role: "user",
        content: question,
      },
    ]);

    try {
      const response = await sendChatMessage(
        selectedDocumentId,
        question
      );

      const answer = Array.isArray(response.answer)
        ? response.answer
          .map((block) => block.text || "")
          .join("\n")
        : response.answer;

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: answer,
          sources: response.sources || [],
          response_time_ms: response.response_time_ms,
        },
      ]);
    } catch (error) {
      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content: `Error: ${error.message}`,
        },
      ]);

      throw error;
    }
  }

  const selectedDocument =
    documents.find(
      (doc) => doc.document_id === selectedDocumentId
    ) || null;

  return (
    <div className="app-shell">
      <TopBar
        apiOnline={apiOnline}
        onMonitoring={() => setActiveView("monitoring")}
      />

      {error && (
        <div className="global-error" role="alert">
          <span>{error}</span>
          <button onClick={() => setError("")}>Dismiss</button>
        </div>
      )}

      {activeView === "monitoring" ? (
        <LangSmithDashboard
          onClose={() => setActiveView("chat")}
        />
      ) : (
        <main className="workspace">
          <DocumentPanel
            documents={documents}
            selectedDocumentId={selectedDocumentId}
            onSelect={setSelectedDocumentId}
            onUpload={handleUpload}
            onDelete={handleDeleteDocument}
            loadingDocuments={loadingDocuments}
          />

          <ChatPanel
            document={selectedDocument}
            messages={messages}
            onSend={handleSendMessage}
          />
        </main>
      )}


    </div>
  );
}