import { useState } from "react";

import "./App.css";
import {
  ApiError,
  askQuestion,
  deleteStoredConversation,
  getStoredTurns,
  listStoredConversations,
  saveStoredConversation,
  saveStoredTurns,
  setConversationId,
  resetConversation,
} from "./api/chatClient";
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

interface StoredTurn {
  question: string;
  result: AskResponse | null;
  error: string | null;
}

const SUGGESTED_QUESTIONS = [
  {
    icon: "fa-solid fa-database",
    tag: "Question 01",
    text: "What does PSAD stand for and why is it used?",
  },
  {
    icon: "fa-solid fa-stethoscope",
    tag: "Question 02",
    text: "For males needing further investigation because of PSA, what is the recommended next diagnostic test?",
  },
  {
    icon: "fa-solid fa-clock",
    tag: "Question 03",
    text: "What PSA testing interval is recommended for males aged 50 to 69 who decide to undergo testing?",
  },
  {
    icon: "fa-solid fa-chart-column",
    tag: "Question 04",
    text: "What should happen when total PSA is 3.0 micrograms per litre or greater in males aged 50 to 69?",
  },
];

function toStoredTurns(turns: Turn[]): StoredTurn[] {
  return turns.map((t) => ({ question: t.question, result: t.result, error: t.error }));
}

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [savedChats, setSavedChats] = useState(listStoredConversations());
  const [currentConvId, setCurrentConvId] = useState<string | null>(null);

  function persistConversation(convId: string, current: Turn[]) {
    const stored = toStoredTurns(current);
    saveStoredTurns(convId, stored);
    const title = current.find((t) => t.result)?.question ?? current[0]?.question ?? "Untitled chat";
    saveStoredConversation({ id: convId, title, createdAt: Date.now() });
    setSavedChats(listStoredConversations());
    setCurrentConvId(convId);
  }

  async function handleAsk(question: string) {
    const id = Date.now();
    setTurns((prev) => [...prev, { id, question, result: null, error: null }]);
    setIsLoading(true);

    try {
      const result = await askQuestion(question);
      setTurns((prev) => {
        const next = prev.map((t) => (t.id === id ? { ...t, result } : t));
        persistConversation(result.conversation_id, next);
        return next;
      });
    } catch (e) {
      const message = e instanceof ApiError ? e.message : "Couldn't reach the assistant. Is the backend running?";
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, error: message } : t)));
    } finally {
      setIsLoading(false);
    }
  }

  function handleNewConversation() {
    // Persist the current chat before starting a fresh one.
    const convId = currentConvId;
    if (convId && turns.length > 0) {
      persistConversation(convId, turns);
    }
    resetConversation();
    setTurns([]);
    setInputValue("");
    setCurrentConvId(null);
  }

  function handleLoadConversation(convId: string) {
    const stored = getStoredTurns(convId) as StoredTurn[];
    setConversationId(convId);
    setTurns(stored.map((t, i) => ({ id: Date.now() + i, question: t.question, result: t.result, error: t.error })));
    setInputValue("");
    setCurrentConvId(convId);
  }

  function handleDeleteChat(convId: string) {
    deleteStoredConversation(convId);
    setSavedChats(listStoredConversations());
    if (convId === currentConvId) {
      resetConversation();
      setTurns([]);
      setCurrentConvId(null);
    }
  }

  function handleSuggested(question: string) {
    setInputValue(question);
    handleAsk(question);
  }

  return (
    <div className="app">
      {/* ================= SIDEBAR ================= */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <span>&lt;/&gt;</span>
            ByteCode
          </div>

          <button className="new-chat-btn" onClick={handleNewConversation} disabled={isLoading}>
            <i className="fa-solid fa-plus" />
            <span>New Chat</span>
          </button>
        </div>

        <div className="history">
          <p className="history-title">CHAT HISTORY</p>

          {savedChats.map((chat) => (
            <div
              key={chat.id}
              className={chat.id === currentConvId ? "chat-item active" : "chat-item"}
              onClick={() => handleLoadConversation(chat.id)}
            >
              <i className="fa-regular fa-message" />
              <span title={chat.title}>{chat.title.slice(0, 40)}</span>
              <button
                className="chat-delete-btn"
                title="Delete chat"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteChat(chat.id);
                }}
              >
                <i className="fa-solid fa-trash" />
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <button className="sidebar-action" onClick={() => {}}>
            <i className="fa-solid fa-gear" />
            <span>Settings</span>
          </button>
        </div>
      </aside>

      {/* ================= MAIN ================= */}
      <main className="main-content">
        <section className="hero">
          <div className="ascii-logo">
            <span className="logo-bracket">[</span>
            <span className="logo-text">ByteCode</span>
            <span className="logo-bracket">]</span>
          </div>

          <p className="subtitle">Ask questions and explore your data with AI.</p>
        </section>

        <section className="suggested-section">
          <div className="section-heading">
            <div>
              <h2>Suggested Questions</h2>
              <p>Explore the knowledge available in ByteCode.</p>
            </div>
          </div>

          <div className="suggested-grid">
            {SUGGESTED_QUESTIONS.map((q, i) => (
              <button
                key={i}
                className="question-card"
                onClick={() => handleSuggested(q.text)}
                disabled={isLoading}
              >
                <div className="question-icon">
                  <i className={q.icon} />
                </div>

                <div className="question-content">
                  <span>{q.tag}</span>
                  <p>{q.text}</p>
                </div>

                <i className="fa-solid fa-arrow-up-right-from-square arrow" />
              </button>
            ))}
          </div>
        </section>

        <section className="conversation">
          <div className="conversation-header">
            <div>
              <span className="status-dot"></span>
              <span>BYTECODE AI</span>
            </div>

            <button className="clear-chat" onClick={handleNewConversation} disabled={isLoading}>
              <i className="fa-solid fa-trash"></i>
              Clear
            </button>
          </div>

          {turns.length === 0 ? (
            <div className="empty-chat">
              <div className="empty-icon">
                <i className="fa-solid fa-sparkles"></i>
              </div>

              <h3>Start a conversation</h3>

              <p>
                Choose a suggested question or ask ByteCode anything about the
                available knowledge.
              </p>
            </div>
          ) : (
            turns.map((turn) => (
              <div key={turn.id} className="turn">
                <div className="message user-message">{turn.question}</div>

                {turn.error ? (
                  <div className="message ai-message">
                    <div className="ai-label">
                      <span className="status-dot"></span>
                      BYTECODE
                    </div>
                    <p className="answer-error">{turn.error}</p>
                  </div>
                ) : !turn.result ? (
                  <div className="message ai-message">
                    <div className="ai-label">
                      <span className="status-dot"></span>
                      BYTECODE
                    </div>
                    <p className="loading-line">Retrieving and grounding an answer…</p>
                  </div>
                ) : turn.result.status === "answered" ? (
                  <div className="message ai-message">
                    <div className="ai-label">
                      <span className="status-dot"></span>
                      BYTECODE
                    </div>
                    <AnswerCard result={turn.result} />
                  </div>
                ) : (
                  <div className="message ai-message">
                    <div className="ai-label">
                      <span className="status-dot"></span>
                      BYTECODE
                    </div>
                    <RefusalCard result={turn.result} />
                  </div>
                )}
              </div>
            ))
          )}
        </section>

        <div className="chat-input-wrapper">
          <ChatInput
            value={inputValue}
            onChange={setInputValue}
            onSubmit={handleAsk}
            isLoading={isLoading}
          />

          <p className="input-note">ByteCode can make mistakes. Verify important information.</p>
        </div>
      </main>
    </div>
  );
}

