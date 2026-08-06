import { useState } from "react";
import {
  describeError,
  grade,
  type GradeResult,
  type QuizQuestion,
} from "../api";

// `questions[i].expected_points` is the answer key. It must never be read
// or rendered by this component before a grade result comes back -- only
// `result.missed_points` (returned post-grading) is ever shown to the user.

export function Quiz({
  sessionId,
  moduleNumber,
  questions,
}: {
  sessionId: string;
  moduleNumber: number;
  questions: QuizQuestion[];
}) {
  const [idx, setIdx] = useState(0);
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState<GradeResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  if (questions.length === 0) return null;
  const q = questions[idx];

  async function submit() {
    setBusy(true);
    setError("");
    try {
      setResult(await grade(sessionId, moduleNumber, idx, answer));
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  }

  function next() {
    setIdx(idx + 1);
    setAnswer("");
    setResult(null);
    setError("");
  }

  return (
    <section className="quiz">
      <h3>
        Check yourself ({idx + 1}/{questions.length})
      </h3>
      <p className="question">{q.question}</p>
      <textarea
        rows={4}
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
        disabled={!!result || busy}
        placeholder="Type your answer…"
      />
      {!result && (
        <button onClick={submit} disabled={busy || !answer.trim()}>
          {busy ? "Grading…" : "Submit"}
        </button>
      )}
      {error && <p className="error">{error}</p>}
      {result && (
        <div className={`verdict ${result.verdict}`}>
          <strong>{result.verdict}</strong>
          <p>{result.feedback}</p>
          {result.missed_points.length > 0 && (
            <ul>
              {result.missed_points.map((p) => (
                <li key={p}>{p}</li>
              ))}
            </ul>
          )}
          {idx + 1 < questions.length && (
            <button onClick={next}>Next question</button>
          )}
        </div>
      )}
    </section>
  );
}
