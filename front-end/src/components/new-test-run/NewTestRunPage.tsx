import React,{ useEffect, useState }  from 'react';
import FilterSelect from '../common/Filters/Filters';
import { AllFilters } from '../../types/Filters';

interface FiltersProps {
  onFilterChange?: (filterType: string, value: string) => void;
}

const NewTestRunPage : React.FC<FiltersProps> = ({ onFilterChange }) => {
    const [filters, setFilters] = useState<AllFilters>({
    domains: [],
    languages: [],
    targets: [],
    plans: [],
  metrics: [],
  });
  const [isLoading, setIsLoading] = useState(true);

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
    return (
        <div>
            <h1>Create new Test Run</h1>
            <p>Configure and start AI Evaluation run</p>
            <div className="filters">
                <div className="filters">
      <FilterSelect
        placeholder="Domain"
        filterType="domain"
        options={filters.domains}
        isLoading={isLoading}
        onChange={onFilterChange}
      />

      <FilterSelect
        placeholder="Language"
        filterType="language"
        options={filters.languages}
        isLoading={isLoading}
        onChange={onFilterChange}
      />

      <FilterSelect
        placeholder="Target"
        filterType="target"
        options={filters.targets}
        isLoading={isLoading}
        onChange={onFilterChange}
      />
      <FilterSelect
        placeholder="Test Plan"
        filterType="plan"
        options={filters.plans}
        isLoading={isLoading}
        onChange={onFilterChange}
      />
      <FilterSelect
        placeholder="Metrics"
        filterType="metrics"
        options={filters.metrics}
        isLoading={isLoading}
        onChange={onFilterChange}
      />
    </div>
            </div>
        </div>
    );
}

export default NewTestRunPage;
