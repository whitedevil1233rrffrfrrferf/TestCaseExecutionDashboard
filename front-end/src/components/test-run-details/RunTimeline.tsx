import React, { useEffect, useState } from "react";
import styles from "./runtimeline.module.css";

/* ===== TYPE ===== */

interface TimelineEvent {
  conversation_id: number;
  metric_name: string;
  prompt_ts: string | null;
  response_ts: string | null;
}

interface Props {
  runName: string;
  hoveredMetric: string | null;
}

/* ===== COLORS ===== */
function getRandomColor() {
  // Generate bright, saturated colors
  const hue = Math.floor(Math.random() * 360); // 0–360°
  const saturation = 70 + Math.random() * 30;  // 70–100%
  const lightness = 50 + Math.random() * 20;   // 50–70%
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}


/* ===== COMPONENT ===== */

const RunTimeline: React.FC<Props> = ({ runName, hoveredMetric }) => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);

  useEffect(() => {
    fetch(`http://localhost:8000/test-runs/${runName}/timeline`)
      .then((res) => res.json())
      .then(setEvents);
  }, [runName]);

  if (events.length === 0) return null;

  /* 🧮 TIME MATH */

  const start = Math.min(
    ...events.map(e => new Date(e.prompt_ts!).getTime())
  );

  const end = Math.max(
    ...events.map(e => new Date(e.response_ts!).getTime())
  );

  const total = end - start;

  const MARKERS = 5;

  const totalSeconds = Math.ceil(total / 1000);

    const timestamps = [
    start, // timeline start
    ...events.map(e => new Date(e.response_ts!).getTime()), // each block end
    end // timeline end
    ];
    const uniqueTimestamps = Array.from(new Set(timestamps)).sort((a, b) => a - b);

    const timeMarkers = uniqueTimestamps.map(ts => Math.floor((ts - start)/1000));
    const uniqueMetrics = Array.from(new Set(events.map(e => e.metric_name)));

    // Assign a random color to each metric
    const METRIC_COLORS: Record<string, string> = {};
    uniqueMetrics.forEach(metric => {
    METRIC_COLORS[metric] = getRandomColor();
    });
  return (
  <div className={styles.timelineCard}>
  <div className={styles.timelineHeader}>
    <h3>Execution Timeline</h3>
    <span className={styles.timelineHint}>
      Hover a metric row to highlight execution
    </span>
  </div>

  <div className={styles.wrapper}>
    <div className={styles.timeline}>
      {events.map((e) => {
        const left =
          ((new Date(e.prompt_ts!).getTime() - start) / total) * 100;

        const width =
          ((new Date(e.response_ts!).getTime() -
            new Date(e.prompt_ts!).getTime()) /
            total) * 100;

        return (
          <div
            key={e.conversation_id}
            className={styles.block}
            style={{
              left: `${left}%`,
              width: `${width}%`,
              opacity:
                hoveredMetric === null
                  ? 0.3
                  : hoveredMetric === e.metric_name
                  ? 1
                  : 0.3,
            }}
          />
        );
      })}
    </div>

    <div className={styles.scale}>
      {uniqueTimestamps.map((ts, i) => (
        <div
          key={i}
          className={styles.scaleItem}
          style={{ left: `${((ts - start) / total) * 100}%` }}
        >
          {(ts - start) / 1000}s
        </div>
      ))}
    </div>
  </div>
</div>
);
};

export default RunTimeline;
