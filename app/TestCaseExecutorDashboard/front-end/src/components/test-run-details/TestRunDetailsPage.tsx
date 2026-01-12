import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import styles from "./TestRunDetails.module.css";
import Modal from "./Modal";
import RunTimeline from "./RunTimeline";
import DetailCard from "../common/DetailCard/DetailCard";
import Filters from "./Filters";
import RunDetailsFilters from "../common/Filters/FiltersRunDet";


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
  score?: number | null;
}

interface RunDetailsResponse {
  summary: RunSummary;
  details: RunDetail[];
}
interface FilterOption {
  filter_name: string;
}

interface AllFilters {
  metrics: FilterOption[];
  statuses: FilterOption[];
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
   const [filtersData, setFiltersData] = useState<AllFilters>({
    metrics: [],
    statuses: [],
  });

  const [activeFilters, setActiveFilters] = useState<{
    metric?: string;
    status?: string;
  }>({});

  const statusMap = (status: string | null | undefined): "COMPLETED" | "RUNNING" | "FAILED" | undefined => {
    if (status === "COMPLETED" || status === "RUNNING" || status === "FAILED") return status;
    return undefined;
  };
   const handleFilterChange = (
    filterType: "metric" | "status",
    value: string
  ) => {
    setActiveFilters((prev) => {
      if (!value) {
        const copy = { ...prev };
        delete copy[filterType];
        return copy;
      }
      return { ...prev, [filterType]: value };
    });
  };

  useEffect(() => {
    fetch("http://localhost:8000/get_all_filters")
      .then((res) => res.json())
      .then((data) => {
        setFiltersData({
          metrics: data.metrics,
          statuses: data.statuses,
        });
      })
      .catch(console.error);
  }, []);

  /* ======================
     FETCH RUN DETAILS
  ====================== */

  useEffect(() => {
    if (!runName) return;

    setLoading(true);
    setError(null);

    const params = new URLSearchParams();
    if (activeFilters.metric) params.append("metric", activeFilters.metric);
    if (activeFilters.status) params.append("status", activeFilters.status);

    fetch(
      `http://localhost:8000/test-runs/${encodeURIComponent(
        runName
      )}?${params.toString()}`
    )
      .then((res) => {
        if (!res.ok) throw new Error(`API ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setSummary(data.summary);
        setDetails(data.details);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [runName, activeFilters]);
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
          <DetailCard
            label="Target"
            value={summary.target ?? "-"}
            icon="bi-bullseye"
          />
          <DetailCard
            label="Domain"
            value={summary.domain ?? "-"}
            icon="bi-globe"
          />
          <DetailCard
            label="Status"
            value={summary.status}
            status={statusMap(summary.status)} // only Status card
            icon="bi-activity"
          />
          <DetailCard
            label="Started At"
            value={new Date(summary.start_ts).toLocaleString()}
            icon="bi-calendar-event"
          />
          <DetailCard
            label="Ended At"
            value={
              summary.end_ts ? new Date(summary.end_ts).toLocaleString() : "-"
            }
            icon="bi-calendar-event"
          />
          <DetailCard
            label="Duration"
            value={durationSeconds !== null ? `${durationSeconds}s` : "-"}
            icon="bi-clock"
          />
      </div>
      </div>
      <RunTimeline runName={summary.run_name} hoveredMetric={hoveredMetric}/>       
      <RunDetailsFilters
        metrics={filtersData.metrics}
        statuses={filtersData.statuses}
        loading={loading}
        activeFilters={activeFilters}
        onFilterChange={handleFilterChange}
      /> 
      {/* ===== DETAILS TABLE ===== */}
      <div className="table-responsive table-container">
  <table className="table table-hover table-bordered align-middle mb-0">
    <thead className="table-light">
      <tr>
        
        <th>Testcase</th>
        <th>Metric</th>
        <th>Plan</th>
        <th>Score</th>
        <th>Status</th>
      </tr>
    </thead>

    <tbody>
      {details.length === 0 ? (
        <tr>
          <td colSpan={6} className="text-center py-4 text-muted">
            No test case details found
          </td>
        </tr>
      ) : (
        details.map((d) => (
          <tr
            key={d.detail_id}
            role="button"
            className="cursor-pointer"
            data-bs-toggle="modal"
            data-bs-target="#conversationModal"
            onClick={() =>
              setSelectedConversationId(Number(d.conversation_id))
            }
            onMouseEnter={() => setHoveredMetric(d.metric_name)}
            onMouseLeave={() => setHoveredMetric(null)}
          >
            
            <td>{d.testcase_name}</td>
            <td>{d.metric_name}</td>
            <td>{d.plan_name}</td>
            
            <td>{d.score === null ? "-" : d.score}</td>
            <td>
              <span
                className={`badge ${
                  d.status === "PASSED"
                    ? "bg-success"
                    : d.status === "FAILED"
                    ? "bg-danger"
                    : "bg-secondary"
                }`}
              >
                {d.status}
              </span>
            </td>
          </tr>
        ))
      )}
    </tbody>
  </table>
</div>

        <Modal conversationId={selectedConversationId} />    
        
    </div>
  );
};

export default RunDetails;
