import React, { use, useEffect, useState } from "react";
import "./TestRunsTable.css";

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

const TestRunsTable: React.FC = () => {
  //Use states
  
  const [runs,setRuns]=useState<TestRun[]>([]);
  const [loading,setLoading]=useState<boolean>(true);

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
    })
    .catch(err => console.error(err))
    .finally(() => setLoading(false));
}, []);
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
          {runs.map(run => (
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
              <td>View</td>
              <td>Report</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default TestRunsTable;
