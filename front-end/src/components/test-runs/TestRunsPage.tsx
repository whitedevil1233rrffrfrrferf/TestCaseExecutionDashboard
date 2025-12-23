import React from "react";
import Header from "./Header";
import Filters from "./Filters";
import TestRunsTable from "./TestRunsTable";
import "./TestRunsPage.css";

const TestRunsPage: React.FC = () => {
  return (
    <div className="page-container">
      <Header />
      <Filters />
      <TestRunsTable />
    </div>
  );
};

export default TestRunsPage;
