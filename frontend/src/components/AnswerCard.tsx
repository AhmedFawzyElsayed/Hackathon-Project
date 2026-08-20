import type { AskResponse } from "../types/chat";
import ClaimList from "./ClaimList";
import MetricsBar from "./MetricsBar";

interface AnswerCardProps {
  result: AskResponse;
}

export default function AnswerCard({ result }: AnswerCardProps) {
  const citedCount = result.claims.filter((c) => c.citation).length;

  return (
    <div className="answer-content">
      <div className="answer-status-row">
        <span className="status-badge status-badge--answered">Answered</span>
        <span className="confidence-badge">confidence: {result.confidence_label}</span>
      </div>

      <p className="answer-summary">{result.answer_summary}</p>

      <ClaimList claims={result.claims} />

      <p className="coverage-line">
        {citedCount}/{result.claims.length} claims cited
        {result.unverified_citations.length > 0 &&
          ` · ${result.unverified_citations.length} unverified citation(s)`}
      </p>

      <MetricsBar metrics={result.metrics} />
    </div>
  );
}