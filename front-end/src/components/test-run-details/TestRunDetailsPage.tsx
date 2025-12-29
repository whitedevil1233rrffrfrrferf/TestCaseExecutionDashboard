import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

/* ======================
   TYPES
====================== */

interface RunSummary {
  run_id: number;
  run_name: string;
  target: string | null;
  domain: string | null;
  status: string;
  start_ts: string;
  end_ts: string | null;
}

interface RunDetail {
  detail_id: number;
  run_name: string;
  testcase_name: string;
  metric_name: string;
  plan_name: string;
  conversation_id: string;
  status: string;
}

interface RunDetailsResponse {
  summary: RunSummary;
  details: RunDetail[];
}

/* ======================
   COMPONENT
====================== */

const RunDetails: React.FC = () => {
  const { runName } = useParams<{ runName: string }>();

  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [details, setDetails] = useState<RunDetail[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runName) {
      setError("Run name missing in URL");
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);

    fetch(`http://localhost:8000/test-runs/${encodeURIComponent(runName)}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`API error: ${res.status}`);
        }
        return res.json();
      })
      .then((data: RunDetailsResponse) => {
        setSummary(data.summary);
        setDetails(data.details);
      })
      .catch((err) => {
        console.error(err);
        setError(err.message);
      })
      .finally(() => setLoading(false));
  }, [runName]);

  /* ======================
     STATES
  ====================== */

  if (loading) return <p>Loading test run...</p>;
  if (error) return <p style={{ color: "red" }}>{error}</p>;
  if (!summary) return <p>No test run found</p>;

  const durationSeconds =
    summary.end_ts
      ? Math.round(
          (new Date(summary.end_ts).getTime() -
            new Date(summary.start_ts).getTime()) / 1000
        )
      : null;

  /* ======================
     UI
  ====================== */

  return (
    <div style={{ padding: "20px" }}>
      {/* ===== SUMMARY ===== */}
      <h2>{summary.run_name}</h2>

      <div style={{ marginBottom: "20px" }}>
        <p><strong>Target:</strong> {summary.target ?? "-"}</p>
        <p><strong>Domain:</strong> {summary.domain ?? "-"}</p>
        <p><strong>Status:</strong> {summary.status}</p>
        <p>
          <strong>Started At:</strong>{" "}
          {new Date(summary.start_ts).toLocaleString()}
        </p>
        <p>
          <strong>Ended At:</strong>{" "}
          {summary.end_ts
            ? new Date(summary.end_ts).toLocaleString()
            : "-"}
        </p>
        <p>
          <strong>Duration:</strong>{" "}
          {durationSeconds !== null ? `${durationSeconds}s` : "-"}
        </p>
      </div>

      {/* ===== DETAILS TABLE ===== */}
      <table
        border={1}
        cellPadding={8}
        cellSpacing={0}
        style={{ width: "100%" }}
      >
        <thead>
          <tr>
            <th>Detail ID</th>
            <th>Testcase</th>
            <th>Metric</th>
            <th>Plan</th>
            <th>Conversation ID</th>
            <th>Status</th>
          </tr>
        </thead>

        <tbody>
          {details.length === 0 ? (
            <tr>
              <td colSpan={6} style={{ textAlign: "center" }}>
                No test case details found
              </td>
            </tr>
          ) : (
            details.map((d) => (
              <tr key={d.detail_id}>
                <td>{d.detail_id}</td>
                <td>{d.testcase_name}</td>
                <td>{d.metric_name}</td>
                <td>{d.plan_name}</td>
                <td>{d.conversation_id}</td>
                <td>{d.status}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
};

export default RunDetails;
