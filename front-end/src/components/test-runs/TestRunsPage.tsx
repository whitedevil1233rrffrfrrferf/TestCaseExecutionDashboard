import React, { useState } from "react";
import Header from "./Header";
import Filters from "./Filters";
import TestRunsTable from "./TestRunsTable";
import "./TestRunsPage.css";

const TestRunsPage: React.FC = () => {
  const [activeFilters, setActiveFilters] = useState<Record<string, string>>({});

  const handleFilterChange = (filterType: string, value: string) => {
    setActiveFilters(prev => ({
      ...prev,
      [filterType]: value
    }));
    // You can implement filtering logic here or pass filters to TestRunsTable
    console.log(`Filter changed: ${filterType} = ${value}`);
  };

  return (
    <div className="page-container">
      <Header />
      <Filters onFilterChange={handleFilterChange} />
      <TestRunsTable filters={activeFilters} />
    </div>
  );
};

export default TestRunsPage;
