import React,{useState} from 'react';
import './NewTestRunPage.css';

// Import only the Bootstrap CSS for the select components

import CustomSelect from './CustomSelect/CustomSelect';

interface RunFormData {
  target: string;
  testPlanId: number | null;
  metric: string;
  maxTestCases: string;
  domain: string;
  language: string;
}

const NewTestRunPage: React.FC = () => {
  // Sample data for dropdowns
  const targets = ['Vaidhya AI', 'Target 2', 'Target 3'];
  const testPlans = ['Plan 1', 'Plan 2', 'Plan 3'];
  const metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];
  const maxTestCases = ['10', '20', '30', '50', '100'];
  const domains = ['E-commerce', 'Healthcare', 'Finance', 'Education'];
  const languages = ['English', 'Spanish', 'French', 'German', 'Chinese'];

  const [formData, setFormData] = useState<RunFormData>({
    target: "",
    testPlanId: null,
    metric: "",
    maxTestCases: "",
    domain: "",
    language: "",
  });

  const handleChange = <K extends keyof RunFormData>(
    key: K,
    value: RunFormData[K]
  ) => {
    setFormData((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const res = await fetch("http://localhost:8000/start-run", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });

    const data = await res.json();
    console.log("Backend response:", data);
  };

  return (
    <div className="new-test-run-container">
      <h1>Create New Test Run</h1>
      <p className="subtitle">Configure and start AI evaluation run</p>
      
      <div className="form-group">
        <label>Test Run Name</label>
        <input 
          type="text" 
          className="form-input" 
          defaultValue="Regression test" 
        />
      </div>

      <form className="filters-container" onSubmit={handleSubmit}>
        <div className="filters-row">
          <div className="filter-item">
            <label>Target</label>
            <CustomSelect
              options={targets}
              defaultText="Select Target"
              onChange={(val) => handleChange("target", val)}
            />
          </div>

          <div className="filter-item">
            <label>Test Plan</label>
            <input
              type="number"
              placeholder="Enter Test Plan ID"
              value={formData.testPlanId ?? ""}
              onChange={(e) =>
                handleChange("testPlanId", Number(e.target.value))
              }
            />
          </div>

          <div className="filter-item">
            <label>Metrics</label>
            <CustomSelect
              options={metrics}
              defaultText="Select Metrics"
              onChange={(val) => handleChange("metric", val)}
            />
          </div>
        </div>

        <div className="filters-row">
          <div className="filter-item">
            <label>Max test cases</label>
            <CustomSelect
              options={maxTestCases}
              defaultText="Select Max"
              onChange={(val) => handleChange("maxTestCases", val)}
            />
          </div>

          <div className="filter-item">
            <label>Domain</label>
            <CustomSelect
              options={domains}
              defaultText="Select Domain"
              onChange={(val) => handleChange("domain", val)}
            />
          </div>

          <div className="filter-item">
            <label>Language</label>
            <CustomSelect
              options={languages}
              defaultText="Select Language"
              onChange={(val) => handleChange("language", val)}
            />
          </div>
        </div>

        <button type="submit" className="start-button">
          Start Run
        </button>
      </form>

      
    </div>
  );
};

export default NewTestRunPage;