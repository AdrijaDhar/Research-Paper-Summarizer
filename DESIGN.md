# Research Paper Summarizer — Design Doc

Status: draft v1 (2026-09-23)

## 1. Vision

Build and ship a faithful, multimodal, multi-document-capable scientific paper
summarizer — not a research prototype, a real end-to-end system: developed,
deployed, used for your own work, and usable by other people. Hard requirements,
non-negotiable across every phase below:

1. **All modalities, single or multi-paper.** Text, tables, equations, figures/charts,
   and citation structure, for one paper or a synthesis across many.
2. **Watermark-free (or as close to zero as possible) generation.** No provider-side
   text watermarking (e.g. SynthID-Text, used by Gemini and — as of Aug 2026 —
   Claude API models) in the output path. See §7a for exactly how this is achieved
   and what it does and doesn't cover.
3. **Actually deployable.** Not just a local script — a system with a real serving
   layer other people can hit (§5 Phase 9, §9).

4. **Advanced by design, not by accident.** This is explicitly a hands-on ML
   systems project, not a wrapper around off-the-shelf tools: you build and train
   your own versions of a discrete diffusion model, a transfer-learning pipeline,
   an RL (GRPO) loop, and a GNN/hypergraph neural encoder, each as a real
   component in the running system, not a toy notebook exercise. See §10 for
   exactly where each technique lives and what "hand-built" means concretely for
   each one.

It also includes one research-grade component — graph-grounded RL alignment —
strong enough to be a publishable contribution on its own (§8).

Two source proposals feed this doc:

- **Doc A ("UGDS")** — an academic architecture survey proposing a Unified
  Graph-Diffusion Summarizer: heterogeneous hypergraph encoding, discrete block
  diffusion generation, transfer learning, and on-policy RL (OPA-DPO + GRPO over
  diffusion trajectories).
- **Doc B ("M3 Max plan")** — a hardware-grounded implementation plan: local-first
  multimodal RAG (parsing → knowledge graph → GraphRAG → LLM generation →
  faithfulness verification), scoped to what a single person can actually run and
  fine-tune on a 128 GB Mac, with specific models, datasets, and benchmarks.

Doc A is the more ambitious *vision*; Doc B is the more honest *feasibility check*.
This doc merges them: keep all four of Doc A's pillars (structure, generation,
transfer learning, alignment) as core, hand-trained components — the revision
here is **not** "drop the ambitious ML and ship a RAG wrapper," it's "scope each
technique's model size and training budget to what a single M3 Max can actually
train, so every pillar gets built rather than skipped." The one piece kept
genuinely infeasible solo is *frontier-scale* work (pretraining an 8B+ diffusion
LM from scratch, full-parameter 4-network PPO) — those get scoped-down local
equivalents (§2, §6) instead of being cut.

## 2. Reconciling the two proposals

| Doc A pillar | Doc A's version | What this project actually builds | Why |
|---|---|---|---|
| Structural encoding | Heterogeneous Hypergraph Transformer (HEGEL/HAESum-style, trained from scratch) | **Hand-built heterogeneous GNN / hypergraph transformer** (PyTorch, the incidence-matrix message-passing math from Doc A §"Structural Encoding" implemented directly — word/sentence/discourse/entity nodes, hyperedges over multi-sentence themes) trained on top of a KG (typed nodes: paper, section, sentence, entity/claim, figure, table, citation), used for extractive salience scoring and retrieval reranking | Full HEGEL/HAESum-scale training is a multi-GPU research program, but a hypergraph transformer at a scoped size (single paper's node/edge count is small — thousands, not millions) trains fine locally. You write and train the real message-passing layers, not just call a graph-retrieval library. Doubles as the reward verifier in the RL row below. |
| Generation | Discrete Block Diffusion LM (BD3LM/Fast-dLLM), pretrained or adapted from an AR checkpoint | **Hand-trained discrete masked diffusion module** (absorbing-state forward process, bidirectional denoising network, the $\mathcal{L}_{\text{MDM}}$ objective from Doc A implemented and trained yourself) at a scoped size — either (a) a small transformer (~100–350M params) trained from scratch on your summarization corpus, or (b) LoRA-adapting Dream-7B/LLaDA-8B into block-diffusion for this task. Used first as the summary structure/skeleton planner (Phase 6), promoted to an alternative full generator path once it's working, benchmarked head-to-head against the AR (Qwen3) path | Evidence in Doc B: frontier-scale dLLMs still trail AR models on long-form prose, and naive discrete diffusion fails on summarization without a semantic-aware noising fix — so this is a real, open research question, not a solved problem to skip. Building it at a scoped size gets you genuine diffusion-LM training experience (forward masking, parallel unmasking, remasking) and an honest ablation, without betting the core product's quality on an unproven path. |
| Transfer learning | Gap-sentence / denoising pretraining objectives + BRIO contrastive fine-tuning | Domain-adaptive continued pretraining (masked-span / gap-sentence-style objective, implemented directly) on an arXiv/PubMed corpus + LoRA/QLoRA task fine-tuning, all local via MLX; BRIO-style contrastive ranking loss added on top once a baseline exists | This is the cheapest, best-evidenced pillar (DAPT is proven; LoRA on Qwen3 fits comfortably in 128 GB) — full hand-implementation of the pretraining objective and the contrastive loss, not just calling a fine-tuning library blindly. |
| Alignment / RL | OPA-DPO (on-policy realignment + DPO) and GRPO over discrete diffusion trajectories | **Hand-implemented GRPO training loop** (rollout sampling, group-relative advantage $A_i = (R_i - \text{mean}(R))/\text{std}(R)$, policy-gradient update — the actual math from Doc A, not a library call) applied to the AR generator with a composite reward: ROUGE/BERTScore + AlignScore/MiniCheck faithfulness + a **KG-grounded citation-match reward** (from the hypergraph encoder above). Once the diffusion module (Phase 6) is trained, the same GRPO loop is extended over its denoising trajectories per Doc A's multi-timestep formulation, as a direct comparison point | OPA-DPO's four-network rollout/critique/realign loop needs infra (a large separate critique model) this project doesn't have — GRPO with a KG-grounded reward gets the same "on-policy, faithfulness-aligned" outcome with one policy network. Extending it to diffusion trajectories once that module exists is the natural advanced follow-on and is the flagship novelty (§8). |
| Extractive pre-filtering | Episodic MDP extractor (MemSum) trained via DQN/policy gradient | **Hand-built MDP extractor**: a small ANN (feedforward/GRU-based policy network over sentence + document-state features, exactly MemSum's state/action/reward formulation) trained via policy gradient, using the hypergraph encoder's node embeddings as input features | This is the concrete "ANN" component — a real trained policy network, not a heuristic. Small enough to train locally in hours; a genuine MDP/RL exercise distinct from the GRPO alignment loop. |

## 3. System architecture (end-to-end pipeline)

```
PDF / arXiv ID / DOI
        │
        ▼
1. INGESTION            job queue, PDF store, arXiv/DOI fetch
        │
        ▼
2. MULTIMODAL PARSING    Docling + Granite-Docling (MLX)  → layout, text, tables, DocTags
                         MinerU2.5                          → hard formulas/tables (fallback for low-confidence pages)
                         GROBID                              → references/citation graph edges (TEI-XML)
                         Qwen-VL (2.5/3)                      → chart & figure semantics, caption cross-check
        │
        ▼
3. KNOWLEDGE GRAPH       typed nodes: paper, section, sentence, entity/claim, figure, table, citation
   CONSTRUCTION          typed edges: cites, extends, contradicts, uses-method, reports-metric, co-refers
        │
        ▼
4. HAND-BUILT GNN /      hypergraph transformer over the KG (node-to-hyperedge routing +
   HYPERGRAPH ENCODER    hyperedge-to-node propagation, trained) → node embeddings
        │
        ▼
5. EXTRACTIVE FILTER     hand-built ANN policy network (MemSum-style MDP, trained via
   (ANN / MDP)           policy gradient on the encoder's node embeddings)
        │
        ▼
6. GENERATION            Path A: Qwen3-35B-A3B (MLX, LoRA + DAPT fine-tuned), AR decoding
                         Path B (Phase 6+): hand-trained discrete block-diffusion module
                                             (scoped size / LoRA-adapted Dream/LLaDA) —
                                             benchmarked against Path A
        │
        ▼
7. FAITHFULNESS &        AlignScore / MiniCheck / SummaC verification
   GRAPH-VERIFIED RL     hand-implemented GRPO, reward = ROUGE/BERTScore + faithfulness +
                         KG citation-match; extended over diffusion trajectories once
                         Path B exists
        │
        ▼
Output: single-doc summary / TLDR / multi-doc related-work synthesis
```

## 4. Component & tech choices

| Component | Choice | Notes |
|---|---|---|
| Parsing | Docling + Granite-Docling-258M (MLX) primary; MinerU2.5 for hard formula/table pages; GROBID for citation/reference metadata; Qwen2.5-VL/Qwen3-VL for charts | All run locally, Apache-2.0 where it matters for redistribution |
| Embeddings / rerank | Qwen3-Embedding, Qwen3-Reranker | Local, 32K context, strong MTEB multilingual results |
| Graph store | Start with an embedded graph (NetworkX/DuckDB-backed) for single-machine dev; revisit Neo4j only if multi-doc scale demands it | Deferred decision — spike in Phase 2 |
| **Hypergraph / GNN encoder** | **Hand-built in PyTorch**: heterogeneous node types, hyperedge message passing (node→hyperedge attention, hyperedge→node propagation), trained on the KG | Small graphs per paper (thousands of nodes) — trains locally in minutes/hours, real GNN implementation work |
| **Extractive filter (ANN/MDP)** | **Hand-built** MemSum-style policy network (feedforward/GRU over sentence + document state), trained via policy gradient | Small model, fast local training, genuine RL-on-an-ANN exercise |
| Generator — Path A (interactive) | Qwen3-35B-A3B MoE, MLX 4-bit, LoRA + DAPT fine-tuned | ~20 GB footprint, best quality/GB, the production-quality path |
| Generator — Path A (batch/high-quality) | gpt-oss-120b | Fits in 128 GB but decodes slowly on Apple Silicon (~3–4 tok/s per field reports) — reserve for offline/batch runs |
| **Generator — Path B (diffusion)** | **Hand-trained discrete masked diffusion**: scoped-size (~100–350M) from-scratch model, or LoRA-adapted Dream-7B/LLaDA-8B into block-diffusion | ~8 GB class at 8B scale, fully local; trained and evaluated as a real alternative path, not a demo |
| Transfer learning | Hand-implemented domain-adaptive continued pretraining objective + LoRA/QLoRA via MLX on an arXiv/PubMed subset; BRIO-style contrastive loss added once a baseline exists | Feasible up to ~32B locally |
| RL | **Hand-implemented GRPO** (rollout, group-relative advantage, policy-gradient update), LoRA, local — applied to Path A first, extended to Path B's denoising trajectories once it exists | Full-parameter RL at scale is out of scope without rented cloud GPUs; the *algorithm* is yours either way |
| Evaluation | ROUGE-1/2/L, BERTScore (reporting standard); AlignScore, SummaC, MiniCheck/Bespoke-MiniCheck-7B (faithfulness, decisive metric); LLM-as-judge (secondary) | Faithfulness metrics gate release quality, not just ROUGE |
| Datasets | arXiv, PubMed (single-doc long); SciTLDR (extreme single-sentence); Multi-XScience (multi-doc related-work synthesis); CharXiv, SciCap (multimodal chart/figure eval) | |
| Licensing | Prefer Apache-2.0 models (Qwen, gpt-oss, Mistral, Phi) for anything public-facing; review Llama/Gemma community license terms before redistribution | |

## 5. Phased roadmap

- **Phase 0 — Scaffolding**: repo structure, MLX/PyTorch environment, dependency pinning, dataset acquisition scripts (arXiv/PubMed/SciTLDR/Multi-XScience).
- **Phase 1 — Multimodal ingestion**: Docling/Granite-Docling + MinerU2.5 + GROBID pipeline; coordinate-preserving chunking (needed later for citation grounding, UI highlighting, and as graph node payloads).
- **Phase 2 — KG + hand-built hypergraph/GNN encoder**: typed-node/edge KG construction; implement and train the heterogeneous hypergraph transformer (node/hyperedge message passing) from scratch; single-doc summarization baseline (Qwen3 zero-shot + graph-based retrieval, no fine-tuning yet). Establish ROUGE/BERTScore/AlignScore baseline numbers.
- **Phase 3 — ANN/MDP extractor**: implement and train the MemSum-style policy network (feedforward/GRU) via policy gradient, using the Phase 2 encoder's node embeddings as features; wire it in as the pre-filter ahead of generation.
- **Phase 4 — Transfer learning**: implement the domain-adaptive continued pretraining objective; run it + LoRA fine-tuning of Qwen3 on arXiv/PubMed/SciTLDR/Multi-XScience; add the BRIO-style contrastive loss; re-measure against the Phase 2 baseline.
- **Phase 5 — Faithfulness layer**: wire AlignScore/MiniCheck as both an eval metric and a reward signal; build the KG-grounded citation-match reward (does each claim's cited node actually support it?) using the Phase 2 encoder.
- **Phase 6 — Hand-implemented GRPO (flagship)**: build the GRPO training loop yourself (rollout sampling, group-relative advantage, policy-gradient update) and run it on the Path A (Qwen3) generator with the Phase 5 composite reward. This is the core novel contribution and flagship result (§8).
- **Phase 7 — Discrete diffusion module (Path B)**: implement discrete masked diffusion from scratch (absorbing-state forward process, denoising network, $\mathcal{L}_{\text{MDM}}$ training objective) at a scoped size (~100–350M from-scratch, or LoRA-adapt Dream-7B/LLaDA-8B into block-diffusion); use it first as a structure/skeleton planner feeding Path A, then train it toward being a full alternative generator; benchmark head-to-head against Path A on ROUGE/AlignScore. Extend the Phase 6 GRPO loop over its denoising trajectories.
- **Phase 8 — Multi-document synthesis**: GNN-routed subgraph retrieval across multiple papers using the Phase 2 encoder; related-work generation; evaluate on Multi-XScience.
- **Phase 9 — Productionization**: web UI + API, job queue, auth, deployment for multi-user access (§9); gpt-oss-120b batch mode for highest-quality offline runs; expose both Path A and Path B generators so users/you can compare.

Phases 2–7 are the "advanced ML" core — each hand-builds one requested paradigm
(GNN/hypergraph, ANN+MDP, transfer learning, RL/GRPO, discrete diffusion) rather
than wiring together off-the-shelf tools. See §10 for the technique-to-phase map.

## 6. Compute budget

**Covered locally (M3 Max, 128 GB)** — this is most of the plan: parsing, embeddings/retrieval, the hand-built hypergraph/GNN encoder (small graphs, trains in minutes/hours), the ANN/MDP extractor, LoRA/QLoRA fine-tuning + domain-adaptive continued pretraining up to ~32B, hand-implemented GRPO on the AR generator (LoRA, small effective parameter count), a scoped discrete diffusion model (~100–350M from scratch, or LoRA-adapting Dream-7B/LLaDA-8B), and gpt-oss-120b/Qwen3-35B-A3B inference.

**Needs rented cloud GPUs, only if you chase frontier scale later**: full-parameter (non-LoRA) RL at 30B+, pretraining a diffusion LM at 7B+ from scratch (start with the scoped local version in Phase 7 first — most of the learning value is there regardless of final parameter count), or scaling the diffusion module past what LoRA-adapting an 8B checkpoint gives you. None of this blocks getting every technique working locally first.

## 7. Risks & open questions

- **Scope management, not scope-cutting**: the full UGDS architecture at *frontier scale* (multi-billion-parameter from-scratch diffusion pretraining, 4-network OPA-DPO) is a multi-team research program. Every pillar is still built — hypergraph transformer, discrete diffusion, GRPO, transfer learning — just at a parameter/data scale that trains on one Mac (§2, §6). If a scoped component turns out too weak to be useful, the fallback is to LoRA-adapt a larger pretrained checkpoint (already the plan for diffusion) rather than cutting the technique.
- **Chart/figure hallucination risk**: numeric claims extracted from figures are unreliable (CharXiv shows even strong VLMs have large reasoning gaps on charts). Every such claim must be cross-checked against table/text data before it's used in a reward computation or shipped in a summary.
- **Graph store choice**: embedded vs. Neo4j — deferred to a Phase 2 spike based on actual multi-doc scale.
- **GROBID ↔ Docling/MinerU2.5 integration**: whether TEI output aligns cleanly with parser chunk coordinates needs a Phase 1 spike before the KG schema is finalized.
- **Watermarking**: addressed as a core design constraint, not a footnote — see §7a.

## 7a. No AI-generation watermarking — how, and what it does and doesn't cover

**How it's achieved.** The entire generation path (Qwen3-35B-A3B, gpt-oss-120b,
LoRA-fine-tuned variants, the optional diffusion module) runs as **local
open-weight inference**, not calls to a provider API. This isn't a workaround —
it's a direct consequence of a design already made for cost, latency, and privacy
reasons (§4). Generation-time text watermarking schemes (SynthID-Text, KGW,
Gumbel-max) work by having the *decoder* bias token probabilities in a detectable
pattern; the mark only exists if whoever runs the model chooses to insert it. When
you hold the weights and run the sampling loop yourself, there is no watermark
because nothing in the stack writes one — this is architecturally different from
"stripping" or "evading" a mark added by someone else.

**What to keep out of the pipeline**, since bringing any of these in reintroduces
a mark you don't control:
- Gemini/Vertex AI API calls — SynthID-Text on by default since 2024.
- Claude API models launched on/after 2026-08-02 — watermark "at launch" per
  Anthropic's Aug 2026 disclosure; earlier models covered by 2026-12-02.
- Any other hosted provider API added later (OpenAI, etc.) — check their current
  watermarking status before wiring it in, since these policies shift with
  regulation.

**What this does and doesn't mean.** No token-level statistical mark in the output
is a real, verifiable property of local open-weight generation — it's not a claim
about undetectability by other means (stylometric classifiers, metadata, etc., none
of which this system needs to defeat, since it isn't impersonating a human writer).
Separately: if this is ever deployed as a public-facing service to users in the EU,
Article 50(2) of the EU AI Act places a machine-readable AI-disclosure obligation
on the *deployer* of a generative system, independent of whether the underlying
model watermarks its output. That's a deployment/compliance decision for you to
make explicitly later (§9), not something resolved by the model choice — flagging
it now so it isn't a surprise at launch.

## 8. Suggested flagship novelty

**Graph-verified GRPO for multimodal multi-document scientific synthesis.**

A heterogeneous knowledge graph (papers, sections, claims, entities, figures,
tables, citations) serves simultaneously as (a) the GraphRAG retrieval substrate
and (b) the verifier for a GRPO faithfulness/citation-grounding reward. This
combines the graph/GNN thread and the RL/GRPO thread from both source documents
into one coherent idea, is directly evaluable on Multi-XScience for multi-document
synthesis, and — unlike Doc A's from-scratch diffusion pretraining or 4-network
OPA-DPO infrastructure — is actually buildable solo on local hardware.

## 9. Deployment — letting other people use it

The end goal isn't a local CLI you run for yourself; it's a served system.

- **Serving.** MLX inference behind a small API server (FastAPI) on the M3 Max
  itself for initial use, fronted by a minimal web UI (upload PDF / paste arXiv
  ID → single- or multi-paper mode → streamed summary with citation highlights
  back to source spans, using the coordinate-preserving chunks from Phase 1).
- **Scaling beyond one machine**, if/when usage outgrows a single Mac: the
  system is stateless per-request except for the KG store and job queue, so it
  can move to a small hosted box later without an architecture change — the
  generator is the only piece that would need re-hosting (another Apple Silicon
  box, or a rented GPU box also running the same open-weight model, never a
  watermarking provider API per §7a).
- **Multi-tenant concerns to decide before opening access**: rate limiting /
  queueing (single M3 Max has finite throughput, especially in gpt-oss-120b batch
  mode), basic auth or invite-based access, and — per §7a — whether an AI-content
  disclosure notice is added in the UI once real third-party users are involved
  (recommended regardless of legal jurisdiction, since it costs nothing and
  matches user expectations; the requirement in this doc is no forced statistical
  watermark baked into the text itself, not "no disclosure anywhere in the product").
- **Not yet decided** (flag for a future revision once Phases 1–6 exist to build
  against): hosting provider/cost, auth provider, and whether multi-paper jobs are
  synchronous or queued for async pickup.

## 10. Learning-goals checklist — where each technique actually lives

Every paradigm requested gets a real, hand-built home in the pipeline, not a
decorative mention:

| Technique | Where it's built | What "hand-built" means here |
|---|---|---|
| **GNN / hypergraph neural networks** | Phase 2 | Implement the node-to-hyperedge and hyperedge-to-node message-passing layers yourself (attention + projection matrices), train on the paper's KG |
| **ANN (classic feedforward/RNN net)** | Phase 3 | MemSum-style extractive policy network, trained via policy gradient over an episodic MDP |
| **Transfer learning** | Phase 4 | Domain-adaptive continued pretraining objective implemented and run on arXiv/PubMed, then LoRA/QLoRA task fine-tuning, then BRIO-style contrastive ranking on top |
| **Reinforcement learning (GRPO)** | Phase 6 (AR generator), extended in Phase 7 (diffusion trajectories) | Rollout sampling, group-relative advantage computation, and the policy-gradient update loop written yourself, not a library call |
| **Diffusion learning** | Phase 7 | Discrete absorbing-state forward process, bidirectional denoising network, and the masked-diffusion training loss ($\mathcal{L}_{\text{MDM}}$) implemented and trained, at a scoped size or via LoRA-adapting an existing dLLM checkpoint |

If at any point a phase feels like it's turning into "just call a library and move
on," that's a signal to slow down and implement the core mechanism by hand before
reaching for the off-the-shelf version — that's the whole point of this scope.
