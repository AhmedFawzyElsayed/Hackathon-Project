import { useState } from "react";

import "./App.css";
import { ApiError, askQuestion } from "./api/chatClient";
import AnswerCard from "./components/AnswerCard";
import ChatInput from "./components/ChatInput";
import RefusalCard from "./components/RefusalCard";
import type { AskResponse } from "./types/chat";

interface Turn {
  id: number;
  question: string;
  result: AskResponse | null;
  error: string | null;
}

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  async function handleAsk(question: string) {
    const id = Date.now();
    setTurns((prev) => [...prev, { id, question, result: null, error: null }]);
    setIsLoading(true);

    try {
      const result = await askQuestion(question);
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, result } : t)));
    } catch (e) {
      const message = e instanceof ApiError ? e.message : "Couldn't reach the assistant. Is the backend running?";
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, error: message } : t)));
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <p className="masthead__eyebrow">PCFA-EDPC-2026-001 · retrieval-augmented</p>
        <h1 className="masthead__title">Clinical RAG Assistant</h1>
        <p className="masthead__subtitle">
          Answers grounded in the 2026 prostate cancer early-detection guideline. Every claim
          traces to a cited section and page — questions about your own results are declined,
          not guessed at.
        </p>
      </header>

      <div className="thread">
        <ChatInput onSubmit={handleAsk} isLoading={isLoading} />

        {turns.length === 0 && (
          <p className="empty-state">Ask something like "What does PSAD stand for?" to get started.</p>
        )}

        {turns
          .slice()
          .reverse()
          .map((turn) => (
            <div key={turn.id} className="question-log">
              <span className="question-log__label">You asked</span>
              <span>{turn.question}</span>

              {turn.error && <p className="chat-form__error">{turn.error}</p>}

              {!turn.error && !turn.result && <p className="loading-line">Retrieving and grounding an answer…</p>}

              {turn.result &&
                (turn.result.status === "answered" ? (
                  <AnswerCard result={turn.result} />
                ) : (
                  <RefusalCard result={turn.result} />
                ))}
            </div>
          ))}
      </div>
    </div>
  );
}
