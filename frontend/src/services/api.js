
const API_BASE_URL = "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(
      data.detail || `Request failed (${response.status})`
    );
  }

  return response.json();
}

export function checkHealth() {
  return request("/api/health");
}

export function getDocuments() {
  return request("/api/documents");
}

export function getDocumentStatus(documentId) {
  return request(
    `/api/documents/${encodeURIComponent(documentId)}/status`
  );
}

export function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  return request("/api/documents/upload", {
    method: "POST",
    body: formData,
  });
}

export function deleteDocument(documentId) {
  return request(
    `/api/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" }
  );
}

export function sendChatMessage(documentId, question) {
  return request("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      document_id: documentId,
      question,
    }),
  });
}


export function getLangSmithStatus() {
  return request("/api/monitoring/langsmith");
}


export function getLangSmithSummary() {
  return request("/api/monitoring/langsmith/summary");
}

export function getLangSmithRuns({ limit = 25, offset = 0, status = "all" } = {}) {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
    status,
  });

  return request(`/api/monitoring/langsmith/runs?${params.toString()}`);
}

export function getLangSmithRunDetail(runId) {
  return request(
    `/api/monitoring/langsmith/runs/${encodeURIComponent(runId)}`
  );
}

// RAGAS evaluation analytics

export function getRagasAnalytics() {
  return request("/api/analytics/ragas");
}