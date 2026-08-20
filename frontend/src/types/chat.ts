// Mirrors backend/app/schemas/chat.py, which mirrors rag_core's frozen
// answer_question() contract. If a field is added on the backend, add it
// here in the same commit — don't let these drift out of sync.

export type AnswerStatus = "answered" | "insufficient_evidence" | "patient_specific_refusal";

export interface Citation {
  document_id: string;
  section: string;
  page: number;
  chunk_id: string;
}

export interface Claim {
  claim_text: string;
  citation: Citation | null;
}

export interface GenerationMetrics {
  faithfulness: number;
  answer_relevance: number;
  hallucination_rate: number;
  context_utilization: number;
}

export interface AskResponse {
  status: AnswerStatus;
  answer_summary: string;
  claims: Claim[];
  citation_coverage: number;
  unverified_citations: string[];
  confidence_label: string;
  conversation_id: string;
  metrics: GenerationMetrics;
}
