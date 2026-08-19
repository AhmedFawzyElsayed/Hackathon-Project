import type { AskResponse } from "../types/chat";

interface RefusalCardProps {
  result: AskResponse;
}

const COPY: Record<string, { label: string; modifier: string }> = {
  insufficient_evidence: { label: "Insufficient evidence", modifier: "insufficient" },
  patient_specific_refusal: { label: "Refused — patient-specific", modifier: "refusal" },
};

export default function RefusalCard({ result }: RefusalCardProps) {
  const meta = COPY[result.status] ?? { label: result.status, modifier: "insufficient" };

  return (
    <div className={`result-card result-card--${meta.modifier}`}>
      <div className="result-card__bar" />
      <div className="result-card__body">
        <div className="result-card__status-row">
          <span className={`status-badge status-badge--${meta.modifier}`}>{meta.label}</span>
        </div>
        <p className="result-card__summary">{result.answer_summary}</p>
      </div>
    </div>
  );
}
