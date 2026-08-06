import { useEffect, useState } from "react";
import { describeError, getModule, type ModuleData } from "../api";
import { AskBox } from "./AskBox";
import { Quiz } from "./Quiz";

export function ModuleView({
  sessionId,
  number,
}: {
  sessionId: string;
  number: number;
}) {
  const [data, setData] = useState<ModuleData | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError("");
    getModule(sessionId, number)
      .then((d) => {
        if (!cancelled) setData(d);
      })
      .catch((err) => {
        // Not swallowed: an unhandled rejection here would both crash
        // silently and leave the pane stuck on "Loading module…" forever.
        if (!cancelled) setError(describeError(err));
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, number]);

  if (error) return <p className="error">Couldn't load this module: {error}</p>;
  if (!data) return <p className="loading">Loading module…</p>;

  return (
    <article className="module">
      <h2>
        {data.number}. {data.title}
      </h2>
      <div className="content">{data.content}</div>
      <Quiz sessionId={sessionId} moduleNumber={data.number} questions={data.quiz} />
      <AskBox sessionId={sessionId} />
    </article>
  );
}
