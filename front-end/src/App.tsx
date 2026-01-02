import { Routes, Route } from "react-router-dom";
import React from 'react';
import logo from './logo.svg';
import './App.css';
import TestRunsPage from './components/test-runs/TestRunsPage';
import TestRunDetails from './components/test-run-details/TestRunDetailsPage';
import NewTestRunPage from "./components/new-test-run/NewTestRunPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<TestRunsPage />} />

      {/* 👇 THIS is the route you're navigating to */}
      <Route path="/test-runs/:runName" element={<TestRunDetails />} />
      <Route path="/create-test-run" element={<NewTestRunPage />} />
    </Routes>
  );
}

export default App;
