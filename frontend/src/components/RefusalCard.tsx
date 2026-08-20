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
    <div className="answer-content">
      <div className="answer-status-row">
        <span className={`status-badge status-badge--${meta.modifier}`}>{meta.label}</span>
      </div>

      <p className="answer-summary">{result.answer_summary}</p>
    </div>
  );
}