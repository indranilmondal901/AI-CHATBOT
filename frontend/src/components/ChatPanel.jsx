
import {useState } from "react";
import { Bot, Send, UserRound } from "lucide-react";
import MessageBubble from "./MessageBubble";

export default function ChatPanel({ document, messages, onSend }) {
  const [question, setQuestion] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  // const messagesEndRef = useRef(null);

  const isReady = document?.status === "ready";

  // useEffect(() => {
  //   messagesEndRef.current?.scrollIntoView({
  //     behavior: "smooth",
  //   });
  // }, [messages, sending]);

  async function handleSubmit(event) {
    event.preventDefault();

    const trimmedQuestion = question.trim();

    if (!trimmedQuestion || !document || sending) {
      return;
    }

    setQuestion("");
    setError("");
    setSending(true);

    try {
      await onSend(trimmedQuestion);
    } catch (err) {
      setError(err.message || "Unable to send your question.");
    } finally {
      setSending(false);
    }
  }

  return (
    <section className="chat-panel">
      <header className="chat-header">
        <div className="chat-title-icon">
          <Bot size={20} />
        </div>

        <div>
          <h2>Knowledge Chat</h2>
          {/* <p>
            {document
              ? `Document: ${document.filename}`
              : "Select a document to begin"}
          </p> */}
        </div>

        {document && (
          <span className={`status-pill ${isReady ? "ready" : ""}`}>
            {isReady ? "Ready" : document.status}
          </span>
        )}
      </header>

      {!document ? (
        <div className="chat-empty">
          <Bot size={38} />
          <h3>Your knowledge assistant</h3>
          <p>Upload or select a document to start asking questions.</p>
        </div>
      ) : !isReady ? (
        <div className="chat-empty">
          <Bot size={38} />
          <h3>Preparing your document</h3>
          <p>
            Chat will be available once ingestion and indexing are complete.
          </p>
        </div>
      ) : (
        <>
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="chat-empty">
                <Bot size={38} />
                <h3>Ask your document</h3>
                <p>
                  Ask a question about the uploaded content. Answers
                  should be based on the retrieved document context.
                </p>
              </div>
            )}

            {messages.map((message) => (
              <div
                className={`message-row ${message.role === "user" ? "user-row" : "assistant-row"
                  }`}
                key={message.id}
              >
                <div className="message-avatar">
                  {message.role === "user" ? (
                    <UserRound size={17} />
                  ) : (
                    <Bot size={17} />
                  )}
                </div>

                <MessageBubble message={message} />
              </div>
            ))}

            {sending && (
              <div className="typing-indicator">
                <Bot size={17} />
                <span>Searching document and generating answer...</span>
              </div>
            )}

            <div /*ref={messagesEndRef}*/ />
          </div>

          {error && (
            <p className="chat-error" role="alert">
              {error}
            </p>
          )}

          <form className="chat-input-area" onSubmit={handleSubmit}>
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question about this document..."
              rows={2}
              maxLength={2000}
              disabled={sending}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey &&
                  !event.nativeEvent.isComposing
                ) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
            />

            <button
              type="submit"
              disabled={!question.trim() || sending}
              aria-label="Send question"
              title="Send question"
            >
              <Send size={18} />
            </button>
          </form>

          <p className="chat-hint">
            Answers may be incomplete. Verify important information against
            the original document.
          </p>
        </>
      )}
    </section>
  );
}