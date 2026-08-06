import { useEffect, useState } from "react";
import { ModuleView } from "./components/ModuleView";
import { SetupForm } from "./components/SetupForm";
import { describeError, getSession, type SessionData } from "./api";
import "./App.css";

export default function App() {
  const [session, setSession] = useState<SessionData | null>(null);
  const [current, setCurrent] = useState(1);
  const [researchGaps, setResearchGaps] = useState<string[] | null>(null);
  const [gapsError, setGapsError] = useState("");

  // POST /sessions doesn't return research_gaps (only GET /sessions/{id}
  // does), so fetch it separately once we have a session. This is
  // supplementary context, not load-bearing for the rest of the UI, so a
  // failure here is logged and shown quietly rather than blocking the page.
  useEffect(() => {
    if (!session) return;
    let cancelled = false;
    getSession(session.session_id)
      .then((detail) => {
        if (!cancelled) setResearchGaps(detail.research_gaps);
      })
      .catch((err) => {
        if (cancelled) return;
        console.error("Failed to load research gaps:", err);
        setGapsError(describeError(err));
      });
    return () => {
      cancelled = true;
    };
  }, [session]);

  if (!session) return <SetupForm onReady={setSession} />;

  return (
    <div className="app">
      <aside>
        <h2>{session.company}</h2>
        <p className="role">{session.role}</p>
        <p className="gap">{session.gap_analysis}</p>
        {researchGaps && researchGaps.length > 0 && (
          <div className="research-gaps">
            <h4>What the research couldn't confirm</h4>
            <ul>
              {researchGaps.map((g) => (
                <li key={g}>{g}</li>
              ))}
            </ul>
          </div>
        )}
        {gapsError && <p className="error small">{gapsError}</p>}
        <ol>
          {session.module_titles.map((t, i) => (
            <li key={t}>
              <button
                className={current === i + 1 ? "active" : ""}
                onClick={() => setCurrent(i + 1)}
              >
                {t}
              </button>
            </li>
          ))}
        </ol>
        <p className="sources">{session.sources_used} sources researched</p>
      </aside>
      <main>
        <ModuleView sessionId={session.session_id} number={current} />
        <nav className="pager">
          <button disabled={current === 1} onClick={() => setCurrent(current - 1)}>
            Previous
          </button>
          <button
            disabled={current === session.module_titles.length}
            onClick={() => setCurrent(current + 1)}
          >
            Next module
          </button>
        </nav>
      </main>
    </div>
  );
}
