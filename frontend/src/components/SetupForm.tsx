import { useState } from "react";
import { createSession, describeError, type SessionData } from "../api";

export function SetupForm({ onReady }: { onReady: (s: SessionData) => void }) {
  const [company, setCompany] = useState("");
  const [role, setRole] = useState("");
  const [resume, setResume] = useState("");
  const [jobUrl, setJobUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    try {
      const session = await createSession(
        company,
        role,
        resume,
        jobUrl || undefined
      );
      onReady(session);
    } catch (err) {
      setError(describeError(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="setup-page">
      <form onSubmit={submit} className="setup">
        <h1>Interview Prep Agent</h1>
        <p className="subtitle">
          Tell it who you're interviewing with and it researches the company,
          finds the gaps in your resume, and builds a 7-module prep plan.
        </p>
        <label>
          Company
          <input
            placeholder="e.g. Stripe"
            value={company}
            onChange={(e) => setCompany(e.target.value)}
            required
            disabled={loading}
          />
        </label>
        <label>
          Role
          <input
            placeholder="e.g. Backend Engineer"
            value={role}
            onChange={(e) => setRole(e.target.value)}
            required
            disabled={loading}
          />
        </label>
        <label>
          Job posting URL (optional)
          <input
            placeholder="https://..."
            value={jobUrl}
            onChange={(e) => setJobUrl(e.target.value)}
            disabled={loading}
          />
        </label>
        <label>
          Resume
          <textarea
            placeholder="Paste your resume text"
            rows={10}
            value={resume}
            onChange={(e) => setResume(e.target.value)}
            required
            disabled={loading}
          />
        </label>
        <button disabled={loading}>
          {loading ? "Researching…" : "Build my prep plan"}
        </button>
        {loading && (
          <p className="hint">
            Searching sources, reflecting on gaps, and writing your plan.
            This runs a full research pass and can take a minute or two —
            don't close this tab.
          </p>
        )}
        {error && <p className="error">{error}</p>}
      </form>
    </div>
  );
}
