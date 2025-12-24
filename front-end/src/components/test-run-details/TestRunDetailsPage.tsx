import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

interface RunDetail {
  run_name: string;
  testcase_name: string;
  metric_name: string;
  plan_name: string;
  conversation_id: number;
  status: string;
  detail_id: number;
}

const RunDetails = () => {
  const { runName } = useParams<{ runName: string }>();

  const [details, setDetails] = useState<RunDetail[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!runName) return;

    fetch(`http://localhost:8000/test-runs/${runName}/details`)
      .then(res => res.json())
      .then(data => setDetails(data))
      .catch(err => console.error(err))
      .finally(() => setLoading(false));
  }, [runName]);

  if (loading) return <p>Loading...</p>;

  return (
    <div style={{ padding: "20px" }}>
      <h2>Run Details: {runName}</h2>

      <table border={1} cellPadding={8} cellSpacing={0}>
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
          {details.map(detail => (
            <tr key={detail.detail_id}>
              <td>{detail.detail_id}</td>
              <td>{detail.testcase_name}</td>
              <td>{detail.metric_name}</td>
              <td>{detail.plan_name}</td>
              <td>{detail.conversation_id}</td>
              <td>{detail.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default RunDetails;
