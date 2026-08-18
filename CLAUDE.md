# Claude Code Project Notes

This repository implements a deterministic local orchestration loop rather than using Claude Code as the
runtime agent. Claude Code may be used to maintain the repository, tests, prompts, and Blender tooling.

Runtime architecture:

1. The first script is generated either by DeepSeek or by the configured GPT endpoint (`initial_generator`).
2. The local process statically validates the script and launches Blender 5.2 in background/offline mode.
3. Blender writes multiple PNG views and a `.blend` file.
4. The GPT iteration agent receives the target spec, immutable toolkit/reference/docs, the exact script snapshot
   that produced the images, and JPEG-base64 render views.
5. In one response, GPT returns a structured three-axis review and chooses `pass`, `revise`, or `retake_views`.
   `revise` returns the complete next Blender script; `retake_views` returns a camera-only script revision. Render
   failures are also repaired by this GPT agent using the exact code and Blender log.

The CLI may inject one human hint from a selected iteration onward. Keep this intervention simple and explicit;
do not add interactive prompts, file watchers, or background control channels.

Do not reintroduce a separate reviewer-to-coder handoff. Do not modify the shared toolkit or reference script
merely to make a generated asset pass. Keep API keys in environment variables. Tests must never make paid
network calls.
