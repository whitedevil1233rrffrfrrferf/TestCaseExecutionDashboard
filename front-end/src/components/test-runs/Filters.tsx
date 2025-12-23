import React from "react";
import "./Filters.css";

const Filters: React.FC = () => {
  const options = ["Target", "Domain", "Status", "Language"];
  return (
    <div className="filters">
      {options.map((opt) => (
        <select key={opt} className="filter">
          <option>{opt}</option>
        </select>
      ))}
    </div>
  );
};

export default Filters;
