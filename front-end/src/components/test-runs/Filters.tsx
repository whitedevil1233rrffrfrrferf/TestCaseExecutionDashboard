import React, { useState, useEffect } from "react";
import "./Filters.css";
import { AllFilters, FilterOption } from "../../types/Filters";
interface FiltersProps {
  onFilterChange?: (filterType: string, value: string) => void;
}

const Filters: React.FC<FiltersProps> = ({ onFilterChange }) => {
  const [filters, setFilters] = useState<AllFilters>({
    domains: [],
    languages: [],
    targets: [],
  });
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    setIsLoading(true);
    fetch("http://localhost:8000/get_all_filters")
      .then((res) => res.json())
      .then((data: AllFilters) => {
        setFilters(data);
        setIsLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching filters:", err);
        setIsLoading(false);
      });
  }, []);
  
  const handleFilterChange = (filterType: string, event: React.ChangeEvent<HTMLSelectElement>) => {
    if (onFilterChange) {
      onFilterChange(filterType, event.target.value);
    }
  };
    return (
    <div className="filters">
      <select 
        className="filter" 
        onChange={(e) => handleFilterChange('domain', e)}
        disabled={isLoading}
      >
        <option value="" disabled selected>Domain</option>
        {filters.domains.map((d) => (
          <option key={d.filter_name} value={d.filter_name}>{d.filter_name}</option>
        ))}
      </select>

      <select 
        className="filter" 
        onChange={(e) => handleFilterChange('language', e)}
        disabled={isLoading}
      >
        <option value="" disabled selected>Language</option>
        {filters.languages.map((l) => (
          <option key={l.filter_name} value={l.filter_name}>{l.filter_name}</option>
        ))}
      </select>
      
      <select 
        className="filter" 
        onChange={(e) => handleFilterChange('target', e)}
        disabled={isLoading}
      >
        <option value="" disabled selected>Target</option>
        {filters.targets.map((t) => (
          <option key={t.filter_name} value={t.filter_name}>{t.filter_name}</option>
        ))}
      </select>
    </div>
  );
};

export default Filters;
