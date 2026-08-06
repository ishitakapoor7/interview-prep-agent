import { useState } from "react";
import { ask, describeError } from "../api";

export function AskBox({ sessionId }: { sessionId: string }) {
  const [q, setQ] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit() {
    setBusy(true);
    setError("");
    try {
      const r = await ask(sessionId, q);
      setAnswer(r.answer);
      setSources(r.sources);
    } catch (err) {
      setError(describeError(err));
      setAnswer("");
      setSources([]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="ask">
      <h3>Ask anything about this company</h3>
      <div className="ask-row">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. how do they make money?"
          onKeyDown={(e) => {
            if (e.key === "Enter" && q.trim() && !busy) submit();
          }}
        />
        <button onClick={submit} disabled={busy || !q.trim()}>
          {busy ? "Thinking…" : "Ask"}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {answer && (
        <div className="answer">
          <p>{answer}</p>
          {sources.length > 0 && (
            <ul>
              {sources.map((s) => (
                <li key={s}>
                  <a href={s} target="_blank" rel="noreferrer">
                    {s}
                  </a>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
