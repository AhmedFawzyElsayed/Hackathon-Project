import type { AskResponse } from "../types/chat";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

let currentConversationId: string | null = null;

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export async function askQuestion(question: string): Promise<AskResponse> {
  const res = await fetch(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      conversation_id: currentConversationId,
    }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail ?? `Request failed with status ${res.status}`;
    throw new ApiError(detail, res.status);
  }

  const data: AskResponse = await res.json();
  currentConversationId = data.conversation_id;
  return data;
}

export function resetConversation(): void {
  currentConversationId = null;
}
