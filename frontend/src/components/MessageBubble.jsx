
export default function MessageBubble({ message }) {
  const isUser = message.role === "user";
  const sources = Array.isArray(message.sources)
    ? message.sources
    : [];

  return (
    <div className={`message-bubble ${isUser ? "user" : "assistant"}`}>
      <div className="message-content">{message.content}</div>

      {!isUser && sources.length > 0 && (
        <details className="message-sources">
          <summary>Sources ({sources.length})</summary>

          {sources.map((item, index) => {
            // Supports both filename strings and source objects.
            const source =
              typeof item === "string"
                ? { source: item }
                : item || {};

            return (
              <div
                className="source-item"
                key={`${source.source || "source"}-${index}`}
              >
                <strong>
                  {source.source || source.filename || "Uploaded document"}
                </strong>

                {source.page != null && (
                  <span> · Page {source.page}</span>
                )}

                {source.chunk_id != null && (
                  <span> · Chunk {source.chunk_id}</span>
                )}
              </div>
            );
          })}
        </details>
      )}

      {!isUser && message.response_time_ms != null && (
        <div className="response-time">
          Response time: {Number(message.response_time_ms).toFixed(0)} ms
        </div>
      )}
    </div>
  );
}