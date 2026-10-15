
import { useEffect, useState } from "react";
import { Activity, Bot, Database } from "lucide-react";
import { getLangSmithStatus } from "../services/api";

export default function TopBar({ apiOnline, onMonitoring }) {
  const [langSmith, setLangSmith] = useState({
    status: "checking",
    message: "Checking LangSmith connection...",
    latest_trace_at: null,
  });

  useEffect(() => {
    let active = true;

    async function checkLangSmith() {
      try {
        const result = await getLangSmithStatus();

        if (active) {
          setLangSmith(result);
        }
      } catch {
        if (active) {
          setLangSmith({
            status: "unreachable",
            message: "Could not reach the monitoring endpoint.",
            latest_trace_at: null,
          });
        }
      }
    }

    checkLangSmith();

    const intervalId = setInterval(
      checkLangSmith,
      30000
    );

    return () => {
      active = false;
      clearInterval(intervalId);
    };
  }, []);

  const langSmithLabels = {
    checking: "Checking...",
    disabled: "Tracing Disabled",
    not_configured: "Setup Required",
    unreachable: "Unreachable",
    connected: "Connected · No Traces",
    tracing: "Traces Available",
  };

  const langSmithIsHealthy = [
    "connected",
    "tracing",
  ].includes(langSmith.status);

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-icon">
          <Bot size={23} />
        </div>

        <div>
          <h1>Budhana AI</h1>
          <p>Document Knowledge Assistant</p>
        </div>
      </div>

      <div className="topbar-status">
        <span
          className={`status-dot ${apiOnline ? "online" : "offline"
            }`}
        />

        <span>
          {apiOnline ? "API Connected" : "API Offline"}
        </span>

        <span className="topbar-divider" />

        <Activity
          size={17}
          className="langsmith-icon"
        />

        <span
          className={`langsmith-label ${langSmithIsHealthy ? "online" : "offline"
            }`}
          title={langSmith.message}
        >
          LangSmith:{" "}
          {langSmithLabels[langSmith.status] ||
            "Unknown"}
        </span>

        <Database
          size={17}
          className="database-icon"
          title="ChromaDB knowledge base"
        />

        <button
          type="button"
          className="monitoring-nav-button"
          onClick={onMonitoring}
          title="Open LangSmith monitoring"
        >
          <Activity size={17} />
          <span>Monitoring</span>
        </button>
      </div>
    </header>
  );
}