
import {
  FileText,
  Trash2,
  CheckCircle2,
  Clock3,
  LoaderCircle,
  AlertCircle,
} from "lucide-react";

import UploadDocument from "./UploadDocument";

const stages = [
  { key: "upload_done", label: "Document uploaded" },
  { key: "chunking_done", label: "Text chunking" },
  { key: "vectorization_done", label: "Embedding generation" },
  { key: "saved_to_chromadb", label: "Saved to ChromaDB" },
];

function ProgressList({ document }) {
  const completedStages = document.stages || {};
  const currentStep = document.current_step || "";

  return (
    <div className="progress-list">
      {stages.map((stage) => {
        const complete = Boolean(completedStages[stage.key]);

        const stageStep = {
          upload_done: "upload",
          chunking_done: "chunking",
          vectorization_done: "vectorization",
          saved_to_chromadb: "saving_to_chromadb",
        }[stage.key];

        const active =
          !complete &&
          document.status === "processing" &&
          currentStep === stageStep;

        return (
          <div className="progress-stage" key={stage.key}>
            {complete ? (
              <CheckCircle2 size={15} className="stage-complete" />
            ) : active ? (
              <LoaderCircle size={15} className="spin" />
            ) : (
              <Clock3 size={15} className="stage-pending" />
            )}

            <span className={complete ? "stage-done-text" : ""}>
              {stage.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function DocumentPanel({
  documents,
  selectedDocumentId,
  onSelect,
  onUpload,
  onDelete,
  loadingDocuments,
}) {
  return (
    <aside className="document-panel">
      <div className="panel-heading">
        <div>
          <h2>Knowledge Base</h2>
          <p>Manage your documents</p>
        </div>

        <span className="document-count">{documents.length}</span>
      </div>

      <div className="upload-wrapper">
        <UploadDocument onUpload={onUpload} />
      </div>

      <div className="document-list-heading">
        <h3>Documents</h3>
        <span>{documents.length} total</span>
      </div>

      {loadingDocuments ? (
        <div className="panel-message">
          <LoaderCircle size={18} className="spin" />
          <span>Loading documents...</span>
        </div>
      ) : documents.length === 0 ? (
        <div className="panel-message empty-documents">
          <FileText size={28} />
          <p>No documents uploaded yet.</p>
          <span>Upload a PDF or TXT file to begin.</span>
        </div>
      ) : (
        <div className="document-list">
          {documents.map((document) => {
            const selected =
              document.document_id === selectedDocumentId;

            const processing = document.status === "processing";
            const ready = document.status === "ready";
            const failed = document.status === "failed";

            return (
              <article
                key={document.document_id}
                className={`document-card ${selected ? "selected" : ""}`}
              >
                <button
                  type="button"
                  className="document-select"
                  onClick={() => onSelect(document.document_id)}
                >
                  <span className="document-icon">
                    <FileText size={19} />
                  </span>

                  <span className="document-info">
                    <strong title={document.filename}>
                      {document.filename}
                    </strong>

                    <span className="document-status">
                      {processing && (
                        <LoaderCircle size={13} className="spin" />
                      )}
                      {ready && (
                        <CheckCircle2 size={13} />
                      )}
                      {failed && (
                        <AlertCircle size={13} />
                      )}

                      {document.status || "unknown"}
                    </span>
                  </span>
                </button>

                <button
                  type="button"
                  className="delete-document-button"
                  title="Delete document"
                  aria-label={`Delete ${document.filename}`}
                  disabled={processing}
                  onClick={() => onDelete(document.document_id)}
                >
                  <Trash2 size={15} />
                </button>

                {processing && <ProgressList document={document} />}

                {failed && (
                  <p className="document-error">
                    {document.error || "Document processing failed."}
                  </p>
                )}
              </article>
            );
          })}
        </div>
      )}
    </aside>
  );
}