import React, { useEffect, useState } from "react";
import "./TestRunsTable.css";
import { useNavigate } from "react-router-dom";

// Define the structure of a test run (for future use)
interface TestRun {
  run_id: number;
  run_name: string;
  target: string;
  status: string;
  start_ts: string;
  end_ts: string | null;
  domain: string;
}

interface TestRunsTableProps {
  filters?: Record<string, string>;
}

const TestRunsTable: React.FC<TestRunsTableProps> = ({ filters = {} }) => {
  const navigate = useNavigate();
  const [runs, setRuns] = useState<TestRun[]>([]);
  const [filteredRuns, setFilteredRuns] = useState<TestRun[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  // Table headers
  const headers = [
    "Run Id", "Run Name", "Target", "Started At","Ended At", 
    "Duration", "Status", "Domain", "View", "Report"
  ];

  // Get Data from backend
  useEffect(() => {
    fetch("http://localhost:8000/get_all_test_runs")
      .then(res => res.json())
      .then((data: TestRun[]) => {
        setRuns(data);
        setFilteredRuns(data);
      })
      .catch(err => console.error("Error fetching test runs:", err))
      .finally(() => setLoading(false));
  }, []);
  
  
  // Apply filters when they change
  useEffect(() => {
    if (Object.keys(filters).length === 0) {
      setFilteredRuns(runs);
      return;
    }
    
    const filtered = runs.filter(run => {
      return Object.entries(filters).every(([key, value]) => {
        if (!value) return true;
        
        switch(key) {
          case 'domain':
            return run.domain.toLowerCase() === value.toLowerCase();
          case 'target':
            return run.target.toLowerCase() === value.toLowerCase();
          case 'language':
            // Assuming language might be in another property or needs special handling
            return true;
          default:
            return true;
        }
      });
    });
    
    setFilteredRuns(filtered);
  }, [filters, runs]);
  return (
    <div className="table-container">
      <table>
        <thead>
          <tr>
            {headers.map((header) => (
              <th key={header}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {loading ? (
            <tr>
              <td colSpan={headers.length}>Loading test runs...</td>
            </tr>
          ) : filteredRuns.length === 0 ? (
            <tr>
              <td colSpan={headers.length}>No test runs match the selected filters</td>
            </tr>
          ) : filteredRuns.map(run => (
            <tr key={run.run_id}>
              <td>{run.run_id}</td>
              <td>{run.run_name}</td>
              <td>{run.target}</td>
              <td>{new Date(run.start_ts).toLocaleString()}</td>
              <td>{run.end_ts ? new Date(run.end_ts).toLocaleString() : "-"}</td>
              <td>
                {run.end_ts
                  ? `${Math.round(
                      (new Date(run.end_ts).getTime() - new Date(run.start_ts).getTime()) / 1000
                    )}s`
                  : "-"}
              </td>
              <td>{run.status}</td>
              <td>{run.domain}</td>
              <td><button className="view-btn" onClick={() => navigate(`/test-runs/${run.run_name}`)}>View</button></td>
              <td><button
  className="report-btn"
  onClick={() => {
    const link = document.createElement("a");
    link.href = `http://localhost:8000/test-runs/${run.run_name}/evaluation-report`;
    link.setAttribute("download", `${run.run_name}-evaluation.xlsx`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }}
>
  Report
</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TestRunsTable;
