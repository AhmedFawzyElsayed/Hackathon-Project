import type { AskResponse } from "../types/chat";

// Same-origin by default (single-container deploys: the backend serves the UI).
// Local dev: set VITE_API_BASE_URL=http://localhost:8000 in frontend/.env.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

let currentConversationId: string | null = null;

const CHATS_KEY = "bytecode_chats";

export interface StoredConversation {
  id: string;
  title: string;
  createdAt: number;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export function listStoredConversations(): StoredConversation[] {
  try {
    return JSON.parse(localStorage.getItem(CHATS_KEY) ?? "[]");
  } catch {
    return [];
  }
}

export function saveStoredConversation(conv: StoredConversation): void {
  const all = listStoredConversations().filter((c) => c.id !== conv.id);
  // Append so conversations keep their place in the list (no reordering).
  all.push(conv);
  localStorage.setItem(CHATS_KEY, JSON.stringify(all.slice(-50)));
}

export function getStoredTurns(id: string): { question: string; result: AskResponse | null }[] {
  try {
    return JSON.parse(localStorage.getItem(`bytecode_chat_turns_${id}`) ?? "[]");
  } catch {
    return [];
  }
}

export function saveStoredTurns(id: string, turns: { question: string; result: AskResponse | null }[]): void {
  localStorage.setItem(`bytecode_chat_turns_${id}`, JSON.stringify(turns));
}

export function deleteStoredConversation(id: string): void {
  const all = listStoredConversations().filter((c) => c.id !== id);
  localStorage.setItem(CHATS_KEY, JSON.stringify(all));
  localStorage.removeItem(`bytecode_chat_turns_${id}`);
}

export function setConversationId(id: string | null): void {
  currentConversationId = id;
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