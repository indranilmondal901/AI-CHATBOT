
import { useState } from "react";
import { FileUp, LoaderCircle } from "lucide-react";

export default function UploadDocument({ onUpload }) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  async function handleFileChange(event) {
    const file = event.target.files?.[0];
    event.target.value = "";

    if (!file) return;

    const extension = file.name.split(".").pop()?.toLowerCase();

    if (!["txt", "pdf"].includes(extension)) {
      setError("Please select a TXT or PDF file.");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setError("File size must not exceed 5 MB.");
      return;
    }

    setError("");
    setUploading(true);

    try {
      await onUpload(file);
    } catch (err) {
      setError(err.message || "Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="upload-section">
      <input
        id="document-file-input"
        type="file"
        accept=".txt,.pdf"
        hidden
        disabled={uploading}
        onChange={handleFileChange}
      />

      <label
        htmlFor="document-file-input"
        className={`upload-button ${uploading ? "upload-disabled" : ""}`}
        aria-disabled={uploading}
      >
        {uploading ? (
          <LoaderCircle size={18} className="spin" />
        ) : (
          <FileUp size={18} />
        )}

        {uploading ? "Uploading..." : "Upload document"}
      </label>

      <p className="helper-text">PDF or TXT · Maximum 5 MB</p>

      {error && <p className="error-text">{error}</p>}
    </div>
  );
}