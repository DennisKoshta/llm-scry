# llm-scry — LLM Interpretability Visualizer — Design Doc

**Status:** Draft v0.1
**Author:** Dennis
**Last updated:** April 2026

---

## 1. Overview

A locally-hosted, browser-based tool that visualizes what is happening inside a transformer language model as it generates text. The user enters a prompt, the model generates token-by-token, and the UI exposes the internal machinery of the generation — logprob distributions, attention patterns, layer-wise predictions (logit lens), and the ability to intervene on the model's computation and observe behavioral differences.

The project has three simultaneous aims:

1. **Learning vehicle.** Build genuine fluency with transformer internals, the TransformerLens ecosystem, and the current state of mechanistic interpretability research.
2. **Useful personal tool.** Produce something Dennis actually reaches for when debugging prompts, understanding model behavior, or exploring interp research papers.
3. **Commercial foundation.** Establish the skills and technical substrate from which a real product — LLM DevTools for teams fine-tuning / deploying models, or behavioral explainability for closed-weight models on Bedrock — could later be built.

This document specifies the first of those three: a concrete, shippable project. Commercial direction is discussed at the end but is explicitly out of scope for the implementation.

---

## 2. Goals and Non-Goals

### Goals

- **G1.** Run any supported open-weight model locally and generate text token-by-token, with full internal state accessible.
- **G2.** Provide a web UI that shows, for any generation: per-token logprob distributions, attention patterns across layers and heads, logit-lens projections at each layer, and residual-stream magnitudes.
- **G3.** Support activation patching: capture internals from run A, re-run the model with a subset of activations replaced by values from A, and visualize the behavioral delta.
- **G4.** Make the system fast enough for interactive exploration on a single developer laptop (M-series Mac or workstation with a consumer GPU) using small-to-mid models.
- **G5.** Structure the codebase such that adding new views or analysis types is straightforward.

### Non-Goals (for v1)

- **NG1.** Supporting closed-weight / API-only models (Claude, GPT-5, Bedrock endpoints). This is the eventual commercial wedge but would force a different architecture.
- **NG2.** Sparse autoencoders / dictionary learning. Interesting, but a full project unto itself.
- **NG3.** Training new models or fine-tuning. The tool consumes models, it does not produce them.
- **NG4.** Multi-user / cloud deployment. Single-user local app only.
- **NG5.** Production-grade auth, observability, billing, etc.

### Explicit non-goal worth calling out

This is not intended to be a published research tool (like TransformerLens itself). It is a UI layer *on top of* TransformerLens. The research library does the heavy lifting; this project adds the interactive visualization that the community has repeatedly tried and mostly abandoned (BertViz, Ecco, Inseq — all exist, all feel dated).

---

## 3. Background and Related Work

The mechanistic interpretability community has converged on a standard Python toolchain (TransformerLens, nnsight, SAELens) but has comparatively little investment in polished interactive UIs. Existing tools fall into three categories:

- **Research libraries with notebook-based visualization.** TransformerLens, nnsight. Powerful, but the visualization is entirely up to the user and lives in Jupyter.
- **Static or near-static viz tools.** BertViz (attention heads), Ecco (token attribution, logit lens), Inseq (attribution methods). Mostly maintained lightly, Jupyter-first, limited interactivity.
- **Live web products.** Neuronpedia is the notable example — a hosted SAE feature browser. Excellent reference for what "interpretability as a product" can look like, but scoped to SAE features, not general internals.

This project sits in a gap: a modern, interactive, web-based UI for day-to-day transformer internals exploration, using TransformerLens as the engine.

---

## 4. High-Level Architecture

Three-tier local architecture:

```
┌─────────────────────┐
│  React + TS Frontend │   Browser
│  (Vite, Tailwind,    │
│   D3 for viz)        │
└──────────┬──────────┘
           │ HTTP + SSE/WebSocket
           │
┌──────────▼──────────┐
│  FastAPI Backend    │   localhost:8000
│  - session mgmt     │
│  - streaming        │
│  - capture/patch API│
└──────────┬──────────┘
           │ in-process
           │
┌──────────▼──────────┐
│  Model Layer        │   same Python process
│  - TransformerLens  │
│  - HookedTransformer│
│  - PyTorch + CUDA   │
└─────────────────────┘
```

Key property: the backend and model layer are **one process**. TransformerLens models are heavy (seconds to load, gigabytes of VRAM); we do not want to pay that cost per request. The FastAPI process holds a loaded model and serves the API.

For model switching, the backend exposes a `POST /model/load` endpoint that swaps the in-memory model. Only one model is loaded at a time in v1.

---

## 5. Component Design

### 5.1 Model Layer

**Library:** [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), Neel Nanda's mechanistic interpretability library. It wraps HuggingFace models in a `HookedTransformer` class that exposes every intermediate activation via a hook system and a standardized naming scheme (`blocks.{n}.attn.hook_pattern`, `blocks.{n}.hook_resid_post`, etc.).

**Supported models (tiered by priority):**

| Tier | Model | Params | Purpose |
|------|-------|--------|---------|
| 1 | GPT-2 small | 124M | Fast iteration during development; canonical interp target. |
| 1 | Pythia-410m / 1.4b | 410M / 1.4B | Clean architecture, Pile-trained, well-studied. |
| 2 | Gemma 2 2B | 2B | Modern architecture, small enough to run comfortably. |
| 2 | Llama 3.2 1B / 3B | 1B / 3B | Modern, capable, ecosystem-standard. |
| 3 | Qwen 2.5 7B | 7B | Stretch target — requires decent GPU. |

Start with Tier 1. Add Tier 2 once the UI is solid. Tier 3 is aspirational.

**Capture strategy.** TransformerLens supports a `run_with_cache` method that captures all activations in a single forward pass into a dictionary. This is *expensive* in memory for larger models but simplest to implement. Strategy:

- v1: `run_with_cache` with `names_filter` to capture only what the current UI view needs.
- v2: selective capture — the frontend declares which activations it wants before running generation, backend captures only those.

### 5.2 Backend (FastAPI)

**Why FastAPI:** Dennis already uses it at DatumSure. Async-native, integrates cleanly with SSE/WebSocket, great DX.

**Core endpoints (sketch):**

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/model/load` | Load a model by name. Returns metadata (n_layers, n_heads, d_model, tokenizer info). |
| `GET` | `/model/info` | Current loaded model metadata. |
| `POST` | `/generate` | Kick off a generation. Returns a `session_id`. |
| `GET` | `/generate/{session_id}/stream` | SSE stream of generation events (see below). |
| `GET` | `/session/{session_id}/attention?layer=X&head=Y` | Attention patterns for a given layer/head. |
| `GET` | `/session/{session_id}/logit_lens?position=P` | Per-layer predictions at a token position. |
| `GET` | `/session/{session_id}/residual?position=P` | Residual stream magnitudes at a position. |
| `POST` | `/patch` | Run a patched generation: base session + activation substitutions. Returns a new session. |
| `GET` | `/sessions` | List saved sessions for comparison. |

**Streaming format (SSE).** Each event is a JSON object with a `type` field:

```json
{"type": "token", "position": 12, "token_id": 464, "token_str": " the", "logprob": -1.23}
{"type": "top_k", "position": 12, "alternatives": [{"token_str": " the", "logprob": -1.23}, ...]}
{"type": "done", "session_id": "abc123", "total_tokens": 50}
```

The UI renders tokens as they stream in. Heavy activations (full attention tensors, residual streams) are fetched on-demand via the session endpoints, not streamed.

**Session storage.** A session is a completed generation plus its captured activations. In v1, sessions are kept in an in-memory LRU cache (say, last 10 sessions). Activation tensors are large but not huge for small models (~tens of MB per session).

For v2, consider SQLite + on-disk tensor files (safetensors or just `torch.save`) for persistence across restarts.

### 5.3 Frontend

**Stack:** React + TypeScript + Vite + Tailwind. For visualization: D3 for custom viz (attention heatmaps, residual magnitude plots) and maybe Plotly or Recharts for straightforward charts.

**Information architecture:** a single-page app with a persistent prompt/generation panel at the top and a tabbed inspector below:

- **Tokens tab:** token-by-token view, each token clickable, showing its logprob and top-k alternatives in a side panel.
- **Attention tab:** heatmap of attention patterns. Dropdowns for layer/head. Toggle between single-head view and layer-aggregated view.
- **Logit Lens tab:** for a selected token position, a table/heatmap showing what the model "would have predicted" if each layer's residual stream were projected to logits directly. This often shows the answer crystallizing partway through the stack.
- **Residual tab:** per-layer magnitudes of the residual stream at a selected position, with component breakdown (attention contribution vs MLP contribution per layer).
- **Patch tab (v2):** UI for selecting activations from one session to copy into another, then running the patched generation and diffing outputs.

**Viz note:** Attention visualization is a solved visual problem — heatmap with axes = sequence positions, color = attention weight. The hard part is *interactivity*: hover to highlight, click to lock, brush to compare, overlay multiple heads. Spend time here; this is where existing tools fall short.

---

## 6. Key Technical Decisions

### 6.1 Why TransformerLens, not raw HuggingFace + hooks

TransformerLens gives us (a) standardized activation names across architectures, (b) batched hook management, (c) `run_with_cache`, and (d) `HookedTransformer.run_with_hooks` for intervention. Writing these from scratch on top of HF is a lot of low-value work. Trade-off: TransformerLens sometimes lags on newest architectures — if we hit that wall, we fall back to raw HF for that specific model.

### 6.2 Why local-only, not a hosted service

Three reasons: (1) GPU costs for a hosted playground are prohibitive for a side project; (2) latency of remote model inference kills the "scrub through attention patterns in real time" UX; (3) open-weight model licenses vary and hosting introduces licensing questions we don't need. Local deployment sidesteps all three.

### 6.3 Why SSE over WebSocket for v1

Generation streaming is fundamentally unidirectional: server sends tokens, client receives them. SSE is simpler to implement in FastAPI (`StreamingResponse` with `text/event-stream`), works over plain HTTP, and handles reconnection gracefully. Upgrade to WebSocket only when we need bidirectional (e.g., interactive patching mid-generation).

### 6.4 Why capture-then-query over stream-everything

Streaming every activation alongside every token is a firehose that the frontend cannot usefully consume. Instead, stream only lightweight per-token metadata (token, logprob, top-k) and let the user pull heavy activations on demand by clicking into views. This keeps the streaming path fast and the UI responsive.

### 6.5 Handling activation tensor sizes

For a 24-layer, 12-head model with seq_len=64 and d_model=768:

- Attention tensors: `24 × 12 × 64 × 64` ≈ 1.2M floats ≈ 5 MB
- Residual stream: `24 × 64 × 768` ≈ 1.2M floats ≈ 5 MB
- MLP hidden (if captured): `24 × 64 × 3072` ≈ 5M floats ≈ 20 MB

Totally manageable. For 7B-class models with longer sequences, we start selective capture and may need to truncate seq_len for interp-heavy runs.

---

## 7. Feature Specification

### 7.1 Token-level logprob view (MVP)

For each generated token, show:

- The token string (with whitespace made visible)
- The token's logprob
- Top-k alternatives (k=10 default) with their logprobs, as a bar or bubble chart
- Cumulative logprob / perplexity across the sequence

Interaction: click a token to pin it; the right-side panel shows its full distribution and alternatives. Scrub over tokens to see logprobs change.

### 7.2 Attention visualization

For a selected (layer, head), render a heatmap:

- Rows = query positions (what is attending)
- Columns = key positions (what is being attended to)
- Color intensity = attention weight

Additional views:

- **Head grid.** Small multiples: one heatmap per head in a layer, helps identify heads with distinctive patterns (induction heads, previous-token heads).
- **Layer aggregation.** Mean attention across heads in a layer, for a coarse view.
- **Token-centric.** For a specific output token, show "where did attention come from" as a bar chart over input positions.

### 7.3 Logit lens

For a selected token position, project the residual stream at every layer through the unembedding matrix to get predicted logits *as if* that layer were the last. Render as:

- A table: rows = layers, columns = top-k predicted tokens per layer
- A heatmap: rows = layers, columns = a chosen vocabulary subset (the final top-k), color = logit value

This view is where the "model forming the answer across layers" phenomenon becomes visible. Often, the answer token isn't even in the top-10 until layer 15 out of 24, then spikes. That's the money shot.

### 7.4 Residual stream magnitudes

Per-layer plot of residual stream norm at a selected position, with stacked bars showing attention-output contribution and MLP-output contribution per layer. Helps identify which layers are "doing work" for a given token.

### 7.5 Activation patching (stretch for MVP, core for v1.1)

Flow:

1. Run generation A with prompt P_A. Capture activations.
2. Run generation B with prompt P_B. Capture activations.
3. In the UI, select a subset of activations from A (e.g., "layer 15 residual stream at position 7").
4. Run a third generation with P_B, but at the forward pass substitute the selected activations from A.
5. Visualize: how did B's output distribution change?

UX: side-by-side diff of B vs B-patched token distributions. This is the interactive version of what papers like ROME do programmatically.

---

## 8. Data Model

### 8.1 Session

```python
class Session:
    id: str
    model_name: str
    prompt: str
    generated_tokens: list[Token]
    cache: ActivationCache  # TransformerLens dict-like
    created_at: datetime
    patches: list[PatchSpec] | None  # None for base sessions

class Token:
    position: int
    token_id: int
    token_str: str
    logprob: float
    top_k: list[tuple[int, str, float]]  # (id, str, logprob)
```

### 8.2 PatchSpec

```python
class PatchSpec:
    source_session_id: str
    target_layer: int
    activation_name: str  # e.g. "resid_post"
    position_range: tuple[int, int] | None  # None = all positions
```

Patches are composable: a patched session stores its patches, and can itself be the source for further patches.

---

## 9. Implementation Phases

### Phase 0 — Foundation (weekend 1)

- Repo scaffold: monorepo with `backend/` (FastAPI, uv-managed) and `frontend/` (Vite + React + TS).
- Install TransformerLens, load GPT-2 small, generate 20 tokens in a script, print top-k at each position.
- Docker Compose file for consistency with Dennis's existing workflow. Backend can run outside Docker during dev, though — GPU passthrough is fiddly.

**Exit criterion:** `curl localhost:8000/generate -d '{"prompt":"The capital of France"}'` returns streamed tokens with logprobs.

### Phase 1 — Token view (weekend 2)

- Frontend scaffold: prompt input, generate button, token-by-token render with streaming.
- Top-k alternatives panel on token click.
- Session list sidebar (in-memory).

**Exit criterion:** You can run generations, see tokens appear live, click any token and see its top-k.

### Phase 2 — Attention view (weekend 3)

- Backend endpoint for attention tensors.
- D3 heatmap component.
- Layer/head selectors.
- Small-multiples head grid view.

**Exit criterion:** You can pick any (layer, head) and see its attention pattern. You can identify induction-head-like patterns by eye.

### Phase 3 — Logit lens (weekend 4)

- Backend: per-layer residual → unembedding projection.
- Frontend: layer × top-k table view.

**Exit criterion:** For a factual prompt like "The Eiffel Tower is in the city of", you can see at which layer "Paris" enters the top-k and when it becomes the argmax.

### Phase 4 — Residual stream + polish (weekend 5)

- Residual magnitude view with attention/MLP decomposition.
- Session save/load to disk (pickle or safetensors).
- UI polish, responsive layout, keyboard shortcuts.

**Exit criterion:** The tool is pleasant enough to actually use daily.

### Phase 5 — Activation patching (weekend 6-7)

- Patch spec UI.
- Backend patching via `run_with_hooks`.
- Diff view between base and patched generations.

**Exit criterion:** You can reproduce a minimal version of a ROME-style intervention (swap a single residual-stream position and observe output change) through the UI alone.

### Phase 6+ — Stretch

- Additional models (Gemma, Llama 3.2).
- Attribution methods (integrated gradients, attention-rollout).
- Session diffing UI.
- Export to shareable static HTML reports.

---

## 10. Risks and Open Questions

### Risks

- **R1 — TransformerLens architectural drift.** New models (Gemma 3, Llama 4) may not be supported by TransformerLens for a while. Mitigation: scope v1 to models with confirmed support; have a raw-hooks fallback path.
- **R2 — Performance on larger models.** 7B-class models may be too slow for the "scrub through generations interactively" UX goal. Mitigation: restrict large models to "analyze a completed generation" mode, not live scrubbing.
- **R3 — Frontend complexity.** Attention viz and logit lens are non-trivial D3. Mitigation: start with ugly-but-functional, iterate. Don't spend weekend 3 on animations.
- **R4 — Scope creep.** There are infinite interpretability views worth building. Mitigation: Phases 0-4 are the MVP. Everything else is Phase 5+. Resist.

### Open Questions

- **Q1.** Do we support batched generation (multiple prompts in one request)? Leaning no for v1 — UX is already complex enough with a single generation.
- **Q2.** How do we handle tokenizer quirks (BPE splits, special tokens, whitespace rendering)? Need a careful token rendering component. Steal from OpenAI's tokenizer visualizer or Tiktokenizer.
- **Q3.** Do we expose a Python API in addition to the web UI? Appealing (notebook users are the core audience), but doubles the surface area. Defer.
- **Q4.** SAE integration — even as a "v2 teaser" feature? Would differentiate sharply from existing tools but adds a whole research direction. Defer to explicit v2 planning.

---

## 11. Product Evolution Path

Deliberately out of scope for implementation, but worth writing down so the architecture doesn't accidentally foreclose options.

**Vector 1 — LLM DevTools for teams.** Generalize from "inspect a single generation" to "debug production model behavior." Add: fine-tune diffing (what changed between base and tuned?), regression testing (did this prompt's behavior change after the last fine-tune?), per-generation forensics for flagged outputs. Customer: teams deploying fine-tuned open-weight models.

**Vector 2 — Behavioral interp for closed-weight models.** Strip the mechanistic views (can't do them on Claude/GPT) and double down on black-box techniques: logprob analysis, counterfactual probing, input-attribution via systematic perturbation. Customer: enterprise teams on Bedrock / Azure who need explainability for compliance. Dennis's Bedrock background is a direct fit.

**Vector 3 — Education / research tool.** License to universities and bootcamps as a teaching aid for transformer architecture courses. Lower ceiling, lower effort, possibly a reasonable side revenue stream.

All three vectors share the same core: a great UI for peering into a generation. Build that well and the commercial direction can be chosen later.

---

## 12. Appendix: References and Reading

**Libraries**
- TransformerLens — https://github.com/TransformerLensOrg/TransformerLens
- nnsight — https://nnsight.net/
- SAELens — https://github.com/jbloomAus/SAELens

**Key reading**
- Anthropic's Transformer Circuits thread — https://transformer-circuits.pub/
- Neel Nanda's "200 Concrete Open Problems in Mechanistic Interpretability"
- "A Mathematical Framework for Transformer Circuits" (Elhage et al.)
- "Locating and Editing Factual Associations in GPT" (ROME paper, Meng et al.)
- "Interpretability in the Wild" (Wang et al.) — the indirect object identification circuit paper

**Prior visualization art**
- BertViz — https://github.com/jessevig/bertviz
- Ecco — https://github.com/jalammar/ecco
- Inseq — https://github.com/inseq-team/inseq
- Neuronpedia — https://www.neuronpedia.org/
