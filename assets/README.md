# Demo assets

`demo.gif` is the README demo. It is recorded against **mock data** — a fictional
filing served by a local mock backend that speaks the real SSE `/chat` contract —
so **no real financial filing appears in the recording**.

## Re-recording

The GIF is produced by running the **real frontend** (`next dev`) against a mock
backend, driving it with a headless browser, and encoding the captured video to GIF
with `ffmpeg`. Recording scaffolding is kept out of the repo; the only committed
artifact is `demo.gif` itself.

To refresh it, point the frontend at a backend that returns the SSE chat contract
(`token` / `citation` / `source` / `agent_step` / `done` events), record the chat
view answering one question, and re-encode:

```bash
ffmpeg -i recording.webm -vf "fps=12,scale=1280:-1:flags=lanczos" demo.gif
```

Keep it mock-only — do not record real filings.
