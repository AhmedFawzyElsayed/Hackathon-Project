import { useState } from "react";
import type { FormEvent } from "react";

interface ChatInputProps {
  onSubmit: (question: string) => void;
  isLoading: boolean;
}

export default function ChatInput({ onSubmit, isLoading }: ChatInputProps) {
  const [value, setValue] = useState("");
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
    setValue("");
  }

  return (
    <form className="chat-form" onSubmit={handleSubmit}>
      <div className="chat-form__row">
        <input
          className="chat-form__input"
          type="text"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            if (validationError) setValidationError(null);
          }}
          placeholder="Ask about the early-detection guideline…"
          disabled={isLoading}
          aria-label="Question for the guideline assistant"
        />
        <button className="chat-form__submit" type="submit" disabled={isLoading}>
          {isLoading ? "Asking…" : "Ask"}
        </button>
      </div>
      {validationError && <p className="chat-form__error">{validationError}</p>}
    </form>
  );
}
