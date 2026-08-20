import type { GenerationMetrics } from "../types/chat";

const LABELS: { key: keyof GenerationMetrics; title: string; label: string }[] = [
  { key: "faithfulness", title: "Faithfulness: claim-to-evidence grounding", label: "Faithfulness" },
  { key: "answer_relevance", title: "Answer relevance: question-to-answer overlap", label: "Relevance" },
  { key: "hallucination_rate", title: "Hallucination rate: fraction of uncited claims", label: "Hallucination" },
  { key: "context_utilization", title: "Context utilization: fraction of evidence chunks cited", label: "Context used" },
];

export default function MetricsBar({ metrics }: { metrics: GenerationMetrics }) {
  return (
    <div className="metrics-bar">
      {LABELS.map(({ key, title, label }) => (
        <span key={key} className="metrics-bar__item" title={title}>
          {label}: {Math.round(metrics[key] * 100)}%
        </span>
      ))}
    </div>
  );
}
