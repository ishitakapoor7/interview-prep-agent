// Thin client for the Interview Prep Agent API. Every field name and index
// base here is taken verbatim from backend/app/main.py: module numbers are
// 1-based, quiz question indices are 0-based.

const BASE = "http://localhost:8000";

export type QuizQuestion = { question: string; expected_points: string[] };

export type ModuleData = {
  number: number;
  title: string;
  content: string;
  quiz: QuizQuestion[];
};

// Returned by POST /sessions.
export type SessionData = {
  session_id: string;
  company: string;
  role: string;
  gap_analysis: string;
  module_titles: string[];
  sources_used: number;
};

// Returned by GET /sessions/{id}. Same shape as SessionData minus session_id,
// plus research_gaps -- what the reflection step noticed was missing.
export type SessionDetail = {
  company: string;
  role: string;
  gap_analysis: string;
  module_titles: string[];
  research_gaps: string[];
  sources_used: number;
};

export type GradeResult = {
  verdict: "correct" | "partial" | "incorrect";
  feedback: string;
  missed_points: string[];
};

export type AskResult = { answer: string; sources: string[] };

/** Thrown for any non-2xx response. `status` lets callers branch on 404 vs
 * 502 vs 422 if they want to; everyone else can just show `message`. */
export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Extract a readable message from a FastAPI error body. `detail` can be a
 * plain string (our own HTTPException calls), a list of validation-error
 * objects (FastAPI's automatic 422 for a malformed body), or missing
 * entirely (a proxy/500 that never reached our handlers) -- so every shape
 * is handled instead of assuming `detail` is always a string. */
async function errorMessage(r: Response): Promise<string> {
  const fallback = `HTTP ${r.status} ${r.statusText}`.trim();
  let text: string;
  try {
    text = await r.text();
  } catch {
    return fallback;
  }
  if (!text) return fallback;

  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return text;
  }

  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = detail.map((d) =>
      d && typeof d === "object" && "msg" in d
        ? String((d as { msg: unknown }).msg)
        : JSON.stringify(d)
    );
    return msgs.join("; ") || fallback;
  }
  if (detail != null) return JSON.stringify(detail);
  return fallback;
}

async function handle<T>(r: Response): Promise<T> {
  if (!r.ok) throw new ApiError(r.status, await errorMessage(r));
  return (await r.json()) as T;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(`${BASE}${path}`, init);
  } catch {
    // fetch() itself rejected: the backend isn't reachable at all (down,
    // wrong port, CORS misconfigured) rather than having answered with an
    // error status.
    throw new ApiError(
      0,
      `Could not reach the backend at ${BASE}. Is it running?`
    );
  }
  return handle<T>(r);
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function get<T>(path: string): Promise<T> {
  return request<T>(path);
}

export const createSession = (
  company: string,
  role: string,
  resume_text: string,
  job_url?: string
) =>
  post<SessionData>("/sessions", {
    company,
    role,
    resume_text,
    job_url: job_url || null,
  });

export const getSession = (sessionId: string) =>
  get<SessionDetail>(`/sessions/${sessionId}`);

export const getModule = (sessionId: string, number: number) =>
  get<ModuleData>(`/sessions/${sessionId}/modules/${number}`);

export const ask = (sessionId: string, question: string) =>
  post<AskResult>(`/sessions/${sessionId}/ask`, { question });

/** `moduleNumber` is 1-based, `questionIndex` is 0-based -- matching the
 * module/quiz indices everywhere else in this client. Do not adjust either
 * one here; the backend does no translation of its own. */
export const grade = (
  sessionId: string,
  moduleNumber: number,
  questionIndex: number,
  answer: string
) =>
  post<GradeResult>(`/sessions/${sessionId}/grade`, {
    module_number: moduleNumber,
    question_index: questionIndex,
    answer,
  });

/** Turn any thrown value into a message safe to render. Components should
 * use this instead of `String(err)` so a non-ApiError (a bug, a thrown
 * non-Error value) still renders something readable instead of
 * "[object Object]" or crashing the render. */
export function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Something went wrong.";
}
