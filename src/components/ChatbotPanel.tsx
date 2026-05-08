import { useMemo, useState } from "react";
import { MessageSquare, RotateCcw, Send, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { sendChatMessage } from "@/lib/api";

interface ChatbotPanelProps {
  runId: string;
  bestModelName: string;
  confidenceLabel: string;
}

interface ChatMessage {
  role: "assistant" | "user";
  content: string;
}

const defaultSuggestions = [
  "Give me a business summary",
  "Which model was selected and why?",
  "Is this safe enough for inventory planning?",
  "What should I restock next?",
  "What cleaning was applied?",
];

const buildIntro = (bestModelName: string, confidenceLabel: string) =>
  `I can explain this upload in simple business terms. The selected model is ${bestModelName}, and the current forecast reliability is ${confidenceLabel}. Ask me for a plain summary, a model explanation, or the safest restocking takeaway.`;

const ChatbotPanel = ({ runId, bestModelName, confidenceLabel }: ChatbotPanelProps) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: buildIntro(bestModelName, confidenceLabel),
    },
  ]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  const suggestions = useMemo(() => defaultSuggestions, []);

  const submitMessage = async (message: string) => {
    const trimmed = message.trim();
    if (!trimmed || sending) {
      return;
    }

    setSending(true);
    setError("");
    setMessages((current) => [...current, { role: "user", content: trimmed }]);
    setInput("");

    try {
      const response = await sendChatMessage(runId, trimmed);
      setMessages((current) => [...current, { role: "assistant", content: response.answer }]);
    } catch (caught) {
      const messageText = caught instanceof Error ? caught.message : "Chat request failed.";
      setError(messageText);
    } finally {
      setSending(false);
    }
  };

  const resetChat = () => {
    setMessages([{ role: "assistant", content: buildIntro(bestModelName, confidenceLabel) }]);
    setError("");
    setInput("");
  };

  return (
    <div className="rounded-xl border border-border bg-card p-6">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
            <MessageSquare className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground">Project Assistant</h3>
            <p className="text-xs text-muted-foreground">Ask for a business summary, model justification, forecast reliability, or stock guidance.</p>
          </div>
        </div>
        <button
          type="button"
          onClick={resetChat}
          className="rounded-lg border border-border px-3 py-2 text-xs text-muted-foreground transition hover:text-foreground"
        >
          <RotateCcw className="mr-1 inline h-3 w-3" />
          Reset
        </button>
      </div>

      <div className="mb-4 flex flex-wrap gap-2">
        {suggestions.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => void submitMessage(suggestion)}
            className="rounded-full border border-border bg-muted/60 px-3 py-1 text-xs text-muted-foreground transition hover:border-primary/40 hover:text-foreground"
          >
            <Sparkles className="mr-1 inline h-3 w-3" />
            {suggestion}
          </button>
        ))}
      </div>

      <div className="max-h-96 space-y-3 overflow-y-auto rounded-lg bg-background/60 p-4">
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={`max-w-[90%] whitespace-pre-wrap rounded-lg px-4 py-3 text-sm leading-6 ${
              message.role === "assistant"
                ? "bg-muted text-foreground"
                : "ml-auto bg-primary text-primary-foreground"
            }`}
          >
            {message.content}
          </div>
        ))}
        {sending && (
          <div className="max-w-[90%] rounded-lg bg-muted px-4 py-3 text-sm text-muted-foreground">
            Preparing a cleaner explanation from the latest analysis...
          </div>
        )}
      </div>

      <form
        className="mt-4 flex gap-3"
        onSubmit={(event) => {
          event.preventDefault();
          void submitMessage(input);
        }}
      >
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Example: Which model was selected and why? or Is this safe enough for inventory planning?"
          className="min-h-[52px] flex-1 resize-none rounded-lg border border-border bg-background px-4 py-3 text-sm text-foreground outline-none transition focus:border-primary"
        />
        <Button type="submit" className="self-end" disabled={sending || !input.trim()}>
          <Send className="mr-2 h-4 w-4" />
          Send
        </Button>
      </form>

      {error && <p className="mt-3 text-sm text-destructive">{error}</p>}
    </div>
  );
};

export default ChatbotPanel;
