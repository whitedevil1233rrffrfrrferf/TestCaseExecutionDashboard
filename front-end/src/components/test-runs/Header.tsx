import React from "react";
import "./Header.css";

const Header: React.FC = () => {
  return (
    <div className="header">
      <h1>Test Runs</h1>
      <div className="header-buttons">
        <button className="orange-btn">Continue test run</button>
        <button className="blue-btn">Create a new test run +</button>
      </div>
    </div>
  );
};

export default Header;
