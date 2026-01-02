import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import styles from "./TestRunDetails.module.css";
import Modal from "./Modal";
import RunTimeline from "./RunTimeline";

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
  const [selectedConversationId, setSelectedConversationId] = useState<number | null>(null);
  const [hoveredMetric, setHoveredMetric] = useState<string | null>(null);
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

  if (loading) return <p className={styles.loading}>Loading test run...</p>;
  if (error) return <p className={styles.error}>{error}</p>;
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
    <div className={styles.container}>
      {/* ===== SUMMARY ===== */}
      <div className={styles.summaryCard}>
        <h1 className={styles.title}>
          {summary.run_name}
        </h1>

        <div className={styles.detailsGrid}>
          <div className={styles.detailCard}>
            <div className={styles.detailLabel}>Target</div>
            <div className={styles.detailValue}>
              {summary.target ?? "-"}
            </div>
          </div>

          <div className={styles.detailCard}>
            <div className={styles.detailLabel}>Domain</div>
            <div className={styles.detailValue}>
              {summary.domain ?? "-"}
            </div>
          </div>

          <div className={styles.detailCard}>
            <div className={styles.detailLabel}>Status</div>
            <div className={`${styles.detailValue} ${
              summary.status === "COMPLETED" ? styles.statusCompleted : 
              summary.status === "RUNNING" ? styles.statusRunning : 
              styles.statusFailed
            }`}>
              {summary.status}
            </div>
          </div>

          <div className={styles.detailCard}>
            <div className={styles.detailLabel}>Started At</div>
            <div className={styles.detailValue}>
              {new Date(summary.start_ts).toLocaleString()}
            </div>
          </div>

          <div className={styles.detailCard}>
            <div className={styles.detailLabel}>Ended At</div>
            <div className={styles.detailValue}>
              {summary.end_ts
                ? new Date(summary.end_ts).toLocaleString()
                : "-"}
            </div>
          </div>

          <div className={styles.detailCard}>
            <div className={styles.detailLabel}>Duration</div>
            <div className={styles.detailValue}>
              {durationSeconds !== null ? `${durationSeconds}s` : "-"}
            </div>
          </div>
        </div>
      </div>
      <RunTimeline runName={summary.run_name} hoveredMetric={hoveredMetric}/>          
      {/* ===== DETAILS TABLE ===== */}
      <div className={styles.tableContainer}>
        <table className={styles.table}>
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
                <td colSpan={6} className={styles.emptyState}>
                  No test case details found
                </td>
              </tr>
            ) : (
              details.map((d) => (
                  <tr key={d.detail_id} 
                      data-bs-toggle="modal"
                      data-bs-target="#conversationModal"
                      onClick={() => setSelectedConversationId(Number(d.conversation_id))}
                      onMouseEnter={() => setHoveredMetric(d.metric_name)}
                      onMouseLeave={() => setHoveredMetric(null)}
                      >
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
        <Modal conversationId={selectedConversationId} />    
        

      </div>
    </div>
  );
};

export default RunDetails;
