import React from "react";
import "./Header.css";
import { Play, Plus } from "lucide-react";
const Header: React.FC = () => {
  return (
    <div className="header">
      <h1>Test Runs</h1>
      
      <div className="header-buttons">
        <button className="orange-btn">
          <Play size={16} />
          Continue
        </button>

        <button className="blue-btn">
          <Plus size={16} />
          New Test Run
        </button>
      </div>
    </div>
  );
};

export default Header;
