import React from "react";
import "./Filters.css";
import {useState,useEffect} from "react";
import {AllFilters, FilterOption} from "../../types/Filters";
const Filters: React.FC = () => {
  const options = ["Target", "Domain", "Status", "Language"];
  const [filters, setFilters] = useState<AllFilters>({
    domains: [],
    languages: [],
    targets: [],
  });
  useEffect(() => {
    fetch("http://localhost:8000/get_all_filters")
      .then((res) => res.json())
      .then((data: AllFilters) => setFilters(data))
      .catch((err) => console.error(err));
  }, []);
    return (
     <div className="filters">
      <select className="filter">
        <option value="" disabled selected>Domain</option>
        {filters.domains.map((d) => (
          <option key={d.filter_name}>{d.filter_name}</option>
        ))}
      </select>

      <select className="filter">
        <option>Language</option>
        {filters.languages.map((l) => (
          <option key={l.filter_name}>{l.filter_name}</option>
        ))}
      </select>
      <select className="filter">
        <option>Target</option>
        {filters.targets.map((t) => (
          <option key={t.filter_name}>{t.filter_name}</option>
        ))}
      </select>
    </div>
  );
};

export default Filters;
