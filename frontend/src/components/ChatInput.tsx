import { useState } from "react";
import type { ChangeEvent, FormEvent } from "react";

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (question: string) => void;
  isLoading: boolean;
}

export default function ChatInput({ value, onChange, onSubmit, isLoading }: ChatInputProps) {
  const [validationError, setValidationError] = useState<string | null>(null);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = value.trim();
    if (!trimmed) {
      setValidationError("Type a question before asking.");
      return;
    }
    setValidationError(null);
    onSubmit(trimmed);
    onChange("");
  }

  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    onChange(e.target.value);
    if (validationError) setValidationError(null);
  }

  return (
    <>
      <form className="chat-input" onSubmit={handleSubmit}>
        <button type="button" className="attach-btn" aria-label="Attach file" onClick={() => {}}>
          <i className="fa-solid fa-paperclip"></i>
        </button>

        <input
          type="text"
          value={value}
          onChange={handleChange}
          placeholder="Ask ByteCode anything..."
          disabled={isLoading}
          aria-label="Question for the guideline assistant"
        />

        <button className="send-btn" type="submit" disabled={isLoading}>
          <span>{isLoading ? "Asking…" : "Send"}</span>
          <i className="fa-solid fa-arrow-up"></i>
        </button>
      </form>

      {validationError && <p className="chat-error">{validationError}</p>}
    </>
  );
}