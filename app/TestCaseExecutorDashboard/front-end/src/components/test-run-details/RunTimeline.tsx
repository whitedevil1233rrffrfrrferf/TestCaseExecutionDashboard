import React, { useEffect, useState } from "react";
import styles from "./runtimeline.module.css";

/* ===== TYPES ===== */

interface TimelineEvent {
  conversation_id: number;
  metric_name: string;
  plan_name: string;
  prompt_ts: string | null;
  response_ts: string | null;
}

interface Props {
  runName: string;
  hoveredMetric: string | null;
  hoveredPlan: string | null;
  onHoverPlan: (planName: string | null) => void;
}

/* ===== COMPONENT ===== */

const RunTimeline: React.FC<Props> = ({ runName, hoveredMetric, hoveredPlan, onHoverPlan }) => {
  const [events, setEvents] = useState<TimelineEvent[]>([]);

  useEffect(() => {
    fetch(`http://localhost:8000/test-runs/${runName}/timeline`)
      .then(res => res.json())
      .then(setEvents);
  }, [runName]);

  if (events.length === 0) return null;

  // Filter events to show only the hovered plan or all plans if none is hovered
  const filteredEvents = hoveredPlan 
    ? events.filter(e => e.plan_name === hoveredPlan)
    : events;

  // Group by plan and sort events by prompt time
  const eventsByPlan = filteredEvents.reduce<Record<string, TimelineEvent[]>>(
    (acc, e) => {
      acc[e.plan_name] ||= [];
      acc[e.plan_name].push(e);
      return acc;
    },
    {}
  );

  // Sort events within each plan by prompt time
  Object.values(eventsByPlan).forEach(planEvents =>
    planEvents.sort(
      (a, b) =>
        new Date(a.prompt_ts!).getTime() -
        new Date(b.prompt_ts!).getTime()
    )
  );

  const planNames = Object.keys(eventsByPlan);
  if (planNames.length === 0) return null;

  return (
    <div className={styles.timelineCard}>
      <div className={styles.timelineHeader}>
        <h3>Execution Timeline{hoveredPlan ? `: ${hoveredPlan}` : ''}</h3>
        <span className={styles.timelineHint}>
          {hoveredPlan ? 'Hover a metric row to highlight execution' : 'Hover over a plan in the table to view its timeline'}
        </span>
      </div>
      {/* HEADER */}

      {/* HORIZONTAL ROW (SCROLLS, STICKY SAFE) */}
      <div className={styles.planRow}>
        {planNames.map((plan, index) => {
          const planEvents = eventsByPlan[plan];

          const start = Math.min(
            ...planEvents.map(e => new Date(e.prompt_ts!).getTime())
          );
          const end = Math.max(
            ...planEvents.map(e => new Date(e.response_ts!).getTime())
          );
          const total = end - start || 1;

          return (
            <React.Fragment key={plan}>
              {/* PLAN BLOCK */}
              <div className={styles.planBlock}>
                <div className={styles.planHeader}>{plan}</div>

                {/* TIMELINE */}
                <div className={styles.timeline}>
                  {planEvents.map(e => {
                    const prompt = new Date(e.prompt_ts!).getTime();
                    const response = new Date(e.response_ts!).getTime();

                    const left = ((prompt - start) / total) * 100;
                    const width = ((response - prompt) / total) * 100;

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
                              : 0.25,
                        }}
                      />
                    );
                  })}
                </div>

                {/* SCALE */}
                <div className={styles.scale}>
                  {[0, 0.25, 0.5, 0.75, 1].map((p, i) => (
                    <div
                      key={i}
                      className={styles.scaleItem}
                      style={{ left: `${p * 100}%` }}
                    >
                      {Math.round((total * p) / 1000)}s
                    </div>
                  ))}
                </div>
              </div>

              {/* DOTTED GAP */}
              {index < planNames.length - 1 && (
                <div className={styles.planConnector}>
                  <span className={styles.gapLabel}>
                    Gap between test plans
                  </span>
                </div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};

export default RunTimeline;
