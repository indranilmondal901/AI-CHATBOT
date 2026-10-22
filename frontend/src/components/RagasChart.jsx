
import { useEffect, useState } from "react";
import {
    ResponsiveContainer,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
} from "recharts";
import { getRagasAnalytics } from "../services/api";
import { X } from "lucide-react";


export default function RagasChart({ onClose }) {
    const [metrics, setMetrics] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");

    useEffect(() => {
        let cancelled = false;

        async function loadMetrics() {
            try {
                setLoading(true);
                setError("");

                const data = await getRagasAnalytics();

                if (cancelled) return;

                setMetrics(
                    Array.isArray(data.metrics) ? data.metrics : []
                );

                if (data.status !== "success") {
                    setError(
                        data.message || "No evaluation results available."
                    );
                }
            } catch (err) {
                if (!cancelled) {
                    setError(
                        err.message || "Failed to load RAGAS metrics."
                    );
                }
            } finally {
                if (!cancelled) {
                    setLoading(false);
                }
            }
        }

        loadMetrics();

        return () => {
            cancelled = true;
        };
    }, []);

    if (loading) {
        return <p>Loading evaluation metrics...</p>;
    }

    if (error && metrics.length === 0) {
        return <p role="alert">{error}</p>;
    }

    const chartData = metrics.filter(
        (metric) =>
            typeof metric.score === "number" &&
            Number.isFinite(metric.score)
    );

    if (chartData.length === 0) {
        return (
            <section>
                <h2>RAGAS Evaluation</h2>
                <p>
                    {error || "No valid RAGAS scores available to display."}
                </p>
            </section>
        );
    }


    return (
        <section
            className="ragas-chart"
            style={{ position: "relative", paddingTop: "8px" }}
        >
            <button
                className="ls-icon-button ragas-close-button"
                onClick={onClose}
                aria-label="Close RAGAS dashboard"
                title="Back to chat"
                type="button"
                style={{
                    position: "absolute",
                    top: "8px",
                    right: "8px",
                    zIndex: 10,
                }}
            >
                <X size={19} />
            </button>

            <h2>RAGAS Evaluation</h2>

            <p>Average scores across evaluated test cases (0–1)</p>

            {error && <p role="alert">{error}</p>}

            <div style={{ width: "100%", height: 320 }}>
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={chartData}
                        margin={{
                            top: 20,
                            right: 20,
                            left: 0,
                            bottom: 45,
                        }}
                    >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                            dataKey="name"
                            angle={-15}
                            textAnchor="end"
                            interval={0}
                            height={60}
                        />
                        <YAxis
                            domain={[0, 1]}
                            ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                        />
                        <Tooltip
                            formatter={(value) => [
                                Number(value).toFixed(3),
                                "Average score",
                            ]}
                        />
                        <Bar
                            dataKey="score"
                            name="Average score"
                            radius={[4, 4, 0, 0]}
                        />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </section>
    );

}
