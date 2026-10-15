
import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Coins,
  RefreshCw,
  Search,
  X,
} from "lucide-react";

import {
  getLangSmithSummary,
  getLangSmithRuns,
  getLangSmithRunDetail,
} from "../services/api";

import "./LangSmithDashboard.css";

const PAGE_SIZE = 25;

function formatNumber(value) {
  if (value === null || value === undefined) return "—";
  return Number(value).toLocaleString();
}

function formatLatency(value) {
  if (value === null || value === undefined) return "—";
  return `${Number(value).toLocaleString()} ms`;
}

function formatDate(value) {
  if (!value) return "—";

  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? String(value)
    : date.toLocaleString();
}

function StatusBadge({ status }) {
  return (
    <span className={`ls-status ls-status-${status || "unknown"}`}>
      {status || "unknown"}
    </span>
  );
}

function MetricCard({ icon: Icon, label, value, helper }) {
  return (
    <article className="ls-metric-card">
      <div className="ls-metric-heading">
        <span>{label}</span>
        <Icon size={19} />
      </div>
      <strong>{value}</strong>
      <small>{helper}</small>
    </article>
  );
}

function JsonSection({ title, value }) {
  return (
    <section className="ls-json-section">
      <h4>{title}</h4>
      <pre>
        {value === null || value === undefined
          ? "No data recorded"
          : JSON.stringify(value, null, 2)}
      </pre>
    </section>
  );
}

export default function LangSmithDashboard({ onClose }) {
  const [summary, setSummary] = useState(null);
  const [runs, setRuns] = useState([]);
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [status, setStatus] = useState("all");
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(false);

  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [detailError, setDetailError] = useState("");

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const [summaryResult, runsResult] = await Promise.all([
        getLangSmithSummary(),
        getLangSmithRuns({
          limit: PAGE_SIZE,
          offset,
          status,
        }),
      ]);

      setSummary(summaryResult);
      setRuns(runsResult.runs || []);
      setHasMore(Boolean(runsResult.has_more));
    } catch (err) {
      setError(err.message || "Unable to load LangSmith monitoring data.");
    } finally {
      setLoading(false);
    }
  }, [offset, status]);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  async function openRun(runId) {
    setDetailLoading(true);
    setDetailError("");
    setSelectedDetail(null);

    try {
      const detail = await getLangSmithRunDetail(runId);
      setSelectedDetail(detail);
    } catch (err) {
      setDetailError(err.message || "Unable to load run details.");
    } finally {
      setDetailLoading(false);
    }
  }

  function changeStatus(nextStatus) {
    setStatus(nextStatus);
    setOffset(0);
    setSelectedDetail(null);
  }

  return (
    <main className="ls-dashboard">
      <header className="ls-dashboard-header">
        <div className="ls-dashboard-title">
          <div className="ls-title-icon">
            <Activity size={23} />
          </div>
          <div>
            <h2>LangSmith Monitoring</h2>
            <p>
              Project: <strong>{summary?.project || "Loading…"}</strong>
            </p>
          </div>
        </div>

        <div className="ls-header-actions">
          <button
            className="ls-button ls-button-secondary"
            onClick={loadDashboard}
            disabled={loading}
          >
            <RefreshCw size={16} />
            Refresh
          </button>

          {onClose && (
            <button
              className="ls-icon-button"
              onClick={onClose}
              aria-label="Close monitoring dashboard"
              title="Back to chat"
            >
              <X size={19} />
            </button>
          )}
        </div>
      </header>

      {error && (
        <div className="ls-error-banner">
          <AlertCircle size={19} />
          <div>
            <strong>Monitoring request failed</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {loading && !summary ? (
        <div className="ls-loading">
          <RefreshCw size={22} className="ls-spin" />
          Loading LangSmith data…
        </div>
      ) : (
        <>
          <section className="ls-metrics-grid">
            <MetricCard
              icon={Activity}
              label="Traces"
              value={formatNumber(summary?.total_runs)}
              helper={summary?.scope || "Recent root runs"}
            />
            <MetricCard
              icon={AlertCircle}
              label="Errors"
              value={formatNumber(summary?.error_count)}
              helper={`${formatNumber(summary?.error_rate_percent)}% of sampled runs`}
            />
            <MetricCard
              icon={Clock3}
              label="Average latency"
              value={formatLatency(summary?.average_latency_ms)}
              helper="Completed sampled runs"
            />
            <MetricCard
              icon={Coins}
              label="Reported tokens"
              value={formatNumber(summary?.total_tokens_reported)}
              helper={`${formatNumber(summary?.runs_with_token_data)} runs reported token data`}
            />
          </section>

          <section className="ls-traces-section">
            <div className="ls-section-heading">
              <div>
                <h3>Trace explorer</h3>
                <p>Choose a run to inspect its recorded data.</p>
              </div>

              <label className="ls-filter">
                <Search size={16} />
                <select
                  value={status}
                  onChange={(event) => changeStatus(event.target.value)}
                  aria-label="Filter traces by status"
                >
                  <option value="all">All statuses</option>
                  <option value="success">Successful</option>
                  <option value="error">Errors</option>
                  <option value="running">Running</option>
                </select>
              </label>
            </div>

            <div className="ls-table-wrap">
              <table className="ls-traces-table">
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Latency</th>
                    <th>Started</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((run) => (
                    <tr
                      key={run.id}
                      className="ls-trace-row"
                      onClick={() => openRun(run.id)}
                      tabIndex={0}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          openRun(run.id);
                        }
                      }}
                    >
                      <td>
                        <strong>{run.name || "Unnamed run"}</strong>
                        <small>{run.id}</small>
                      </td>
                      <td>{run.run_type || "—"}</td>
                      <td>
                        <StatusBadge status={run.status} />
                      </td>
                      <td>{formatLatency(run.latency_ms)}</td>
                      <td>{formatDate(run.start_time)}</td>
                    </tr>
                  ))}

                  {!runs.length && !loading && (
                    <tr>
                      <td colSpan="5" className="ls-empty">
                        No runs found for this filter.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <footer className="ls-pagination">
              <span>
                Showing {runs.length ? offset + 1 : 0}–
                {offset + runs.length} in this filtered view
              </span>

              <div>
                <button
                  className="ls-button ls-button-secondary"
                  disabled={offset === 0 || loading}
                  onClick={() => setOffset((current) =>
                    Math.max(0, current - PAGE_SIZE)
                  )}
                >
                  <ChevronLeft size={16} />
                  Previous
                </button>

                <button
                  className="ls-button ls-button-secondary"
                  disabled={!hasMore || loading}
                  onClick={() => setOffset((current) =>
                    current + PAGE_SIZE
                  )}
                >
                  Next
                  <ChevronRight size={16} />
                </button>
              </div>
            </footer>
          </section>

          <p className="ls-data-note">
            Metrics reflect a recent sample, not necessarily the entire
            LangSmith project. Missing token or cost data means the provider
            did not report it for those runs.
          </p>
        </>
      )}

      {(detailLoading || detailError || selectedDetail) && (
        <div
          className="ls-detail-backdrop"
          onClick={() => {
            setSelectedDetail(null);
            setDetailError("");
          }}
        >
          <aside
            className="ls-detail-panel"
            onClick={(event) => event.stopPropagation()}
          >
            <header className="ls-detail-header">
              <div>
                <h3>Run details</h3>
                <p>Inputs, outputs, metadata and child runs</p>
              </div>
              <button
                className="ls-icon-button"
                onClick={() => {
                  setSelectedDetail(null);
                  setDetailError("");
                }}
                aria-label="Close run details"
              >
                <X size={19} />
              </button>
            </header>

            {detailLoading && (
              <div className="ls-loading">
                <RefreshCw size={20} className="ls-spin" />
                Loading run details…
              </div>
            )}

            {detailError && (
              <div className="ls-error-banner">{detailError}</div>
            )}

            {selectedDetail?.run && (
              <div className="ls-detail-content">
                <h3>{selectedDetail.run.name || "Unnamed run"}</h3>
                <p className="ls-detail-id">{selectedDetail.run.id}</p>

                <div className="ls-detail-meta">
                  <span>
                    <strong>Status</strong>
                    <StatusBadge status={selectedDetail.run.status} />
                  </span>
                  <span>
                    <strong>Type</strong>
                    {selectedDetail.run.run_type || "—"}
                  </span>
                  <span>
                    <strong>Latency</strong>
                    {formatLatency(selectedDetail.run.latency_ms)}
                  </span>
                  <span>
                    <strong>Started</strong>
                    {formatDate(selectedDetail.run.start_time)}
                  </span>
                </div>

                {selectedDetail.run.error && (
                  <div className="ls-error-banner">
                    {selectedDetail.run.error}
                  </div>
                )}

                <JsonSection
                  title="Inputs"
                  value={selectedDetail.run.inputs}
                />
                <JsonSection
                  title="Outputs"
                  value={selectedDetail.run.outputs}
                />
                <JsonSection
                  title="Metadata"
                  value={selectedDetail.run.metadata}
                />
                <JsonSection
                  title="Extra"
                  value={selectedDetail.run.extra}
                />
                <JsonSection
                  title="Events"
                  value={selectedDetail.run.events}
                />
                <JsonSection
                  title={`Child runs (${selectedDetail.child_runs?.length || 0})`}
                  value={selectedDetail.child_runs}
                />
              </div>
            )}
          </aside>
        </div>
      )}
    </main>
  );
}