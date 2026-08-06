# Frontend

Vite + React + TypeScript client for the Interview Prep Agent API. See the
repository root [README.md](../README.md) for what this project does, how to
run the backend and frontend together, and the eval methodology.

## Quick start

```bash
npm install
npm run dev
```

Expects the backend running at `http://localhost:8000` (see the root README).

## Layout

- `src/api.ts` — typed client for every backend route, plus error handling
  shared by all components.
- `src/App.tsx` — top-level session state and module navigation.
- `src/components/SetupForm.tsx` — company/role/resume intake, kicks off
  `POST /sessions`.
- `src/components/ModuleView.tsx` — fetches and renders one lesson module.
- `src/components/Quiz.tsx` — per-module quiz flow; grades answers via
  `POST /sessions/{id}/grade`.
- `src/components/AskBox.tsx` — free-form Q&A via `POST /sessions/{id}/ask`.
