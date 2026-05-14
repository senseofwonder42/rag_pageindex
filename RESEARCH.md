# RESEARCH — PageIndex Pipeline Improvement Findings

Audit of the vendored PageIndex pipeline ([src/rag_pageindex/pageindex/](src/rag_pageindex/pageindex/)) with a focus on **simplification, robustness on non-book documents, and tree precision**. The goal is not yet to land changes — only to map out the surface area and propose concrete levers.

---

## 1. Current pipeline at a glance

Entry point: [pipeline.py:apage_index](src/rag_pageindex/pageindex/pipeline.py#L62) → [tree/builder.py:tree_parser](src/rag_pageindex/pageindex/tree/builder.py#L451).

```
read_pages  ──►  check_toc  ──►  meta_processor(mode)  ──►  add_preface  ──►
                                  │                                            
                                  ├── A: process_toc_with_page_numbers         
                                  ├── B: process_toc_no_page_numbers           
                                  └── C: process_no_toc                        
                                  
            ──►  check_title_appearance_in_start_concurrent  ──►  post_processing  ──►
            ──►  process_large_node_recursively (recursive subdivide)  ──►
            ──►  write_node_id  ──►  add_node_text  ──►
            ──►  generate_summaries_for_structure (per node)  ──►
            ──►  generate_doc_description
```

`meta_processor` ([tree/builder.py:189-377](src/rag_pageindex/pageindex/tree/builder.py#L189-L377)) is the *central state machine*: each path produces a TOC; verification scores it; if accuracy is in a "fixable" band it tries text fixers, then VLM fixers; otherwise it **falls through to the next mode**, throwing away the work so far.

### Worst-case LLM-call count for one *N*-page doc with *K* TOC items

| Stage | Calls | Notes |
|---|---:|---|
| `toc_detector_single_page` | up to `toc_check_page_num` | parallel, default 4 |
| `detect_page_index` | 1+ | grows if first TOC page-list lacks page numbers |
| `toc_transformer` | 1 | single structured call; entire TOC in one shot |
| `toc_index_extractor` | 1 | sample of 8 entries vs. main-content prefix |
| `process_none_page_numbers` | 1 per item without `physical_index` | parallel, but K of them |
| `verify_toc` | **K** (sample_n=None) | one per TOC item |
| `fix_incorrect_toc_with_retries` | up to 2 × incorrect (×2 internal calls) | `single_toc_item_index_fixer` + optional `check_title_appearance` |
| `check_title_appearance_in_start_concurrent` | K − (already-cached) | another full pass |
| `process_large_node_recursively` | recursive Path-C per oversized node | can chain deeper passes |
| `generate_summaries_for_structure` | one per node (≈ 2K for a balanced tree) | leaves + parents |
| `generate_doc_description` | 1 | |

A 200-page, 50-section book ⇒ **~250-400 LLM calls** in the happy path, **~600+** if cascades trigger. This is the source of the "large quantity of token in certain case" the user describes.

---

## 2. Major problems & root causes

### 2.1 Pipeline complexity & wasted work

- **Cascade discards verified work.** When Path A's verification accuracy is below `pageindex_vision_fallback_threshold` (0.6), `meta_processor` recurses into Path B with the *raw* `toc_content` again. The K items already located by Path A — many of which are likely correct — are thrown away.  
  Reference: [tree/builder.py:323-343](src/rag_pageindex/pageindex/tree/builder.py#L323-L343).
- **`verify_toc` is doubly expensive.** With `sample_n=None` (the call site at [tree/builder.py:289](src/rag_pageindex/pageindex/tree/builder.py#L289)), every TOC entry triggers a `check_title_appearance` LLM call. Then `check_title_appearance_in_start_concurrent` runs another full pass for items that weren't cached during verify.
- **Path C is sequential.** `process_no_toc` iterates `_generate_toc_continue` group-by-group ([toc/page_mapping.py:413-431](src/rag_pageindex/pageindex/toc/page_mapping.py#L413-L431)), each call carrying the **growing JSON** of all prior items. This both serializes wall-clock and balloons prompt size on long docs.
- **`toc_transformer` is single-shot.** If the TOC text doesn't fit the structured-output cap, the call raises, the whole stage is marked failed, and the cascade falls through — even though `toc_to_json_continue.j2` exists for exactly this case but is **never called** in the current codebase.
- **Hidden duplication of `appear_start` queries.** Same prompt schema (`check_title_appearance`) is used both in `verify_toc` and in `check_title_appearance_in_start_concurrent`. They could be merged into a single fan-out.

### 2.2 Robustness on non-book documents

The pipeline assumes a "book-like" model: contiguous prose, a TOC, hierarchical sections. This fails or wastes tokens on:

- **Slide decks / PowerPoints**: each page is a slide. There's no TOC; Path C invents a hierarchy that does not correspond to slide grouping. Output tree has random 1.x.x indices that mislead retrieval.
- **Financial reports / earnings decks**: heavy mix of tables, charts, footnotes. Text extraction returns lots of numbers with weak heading semantics. `toc_detector_single_page` often returns `"yes"` for a "Highlights" or "Key Figures" page (Path A's prompt explicitly warns against this, but small models still get it wrong) — leading to a Path-A run that fails verification, cascading into Path B and C.
- **Single-table documents / one-pagers**: no hierarchy exists. Pipeline still runs all detection + tree-build calls and returns either an empty tree or a fabricated one.
- **Scanned PDFs**: PyPDF2 extraction yields near-empty text. The pipeline silently produces 0 tokens per page, returns empty TOC, and a degenerate tree. The VLM fallback is gated on text-accuracy thresholds, so scanned docs **do not automatically route to VLM** unless the text path manages to produce a sub-threshold-but-non-zero result.
- **Academic papers**: explicitly called out in [check_toc.j2](src/rag_pageindex/pageindex/prompts/check_toc.j2) ("academic papers almost never contain a TOC") — yet the pipeline still spends `toc_check_page_num` LLM calls per paper to determine there is no TOC, and then runs Path C across the full text. A document-type classifier up front would short-circuit this.

### 2.3 Precision issues in tree extraction

- **`list_to_tree` drops nodes on gappy structure indices.** [tree/builder.py:29-77](src/rag_pageindex/pageindex/tree/builder.py#L29-L77): if the LLM emits `1`, `1.2` (skipping `1.1`), `1.2` is *re-parented to the root list* because its parent `"1.1"` is not in the `nodes` map. Common with small models that miss a heading.
- **`appear_start`-based end-index assignment is fragile.** `post_processing` uses `appear_start == "yes"` to subtract 1 from the next item's start. If `appear_start` is wrong (false negative), end-index ≥ next start-index — pages are double-counted across siblings.
- **`add_preface_if_needed` is a blind insert.** [tree/builder.py:80-103](src/rag_pageindex/pageindex/tree/builder.py#L80-L103) inserts a "Preface" node whenever the first physical_index > 1. Often wrong: PDFs commonly have blank covers, copyright pages, or a Foreword. The node has no title context, hurting downstream retrieval.
- **`verify_toc` early-exit on "last item in front half"** ([toc/verification.py:138-139](src/rag_pageindex/pageindex/tree/verification.py#L138)) returns `0.0` accuracy if `last_physical_index < len(pages)/2`. Correct for "TOC accidentally captured as content" but spuriously triggers on appendix-only books or where the final chapter ends mid-doc.
- **Random sampling in `verify_toc`** when `sample_n` is set: no seed, so two runs over the same doc give different cascade outcomes.
- **`count_tokens` is `len(text)//4`** ([llm/openai_compatible_client.py:17,234](src/rag_pageindex/pageindex/llm/openai_compatible_client.py#L17)). Real tokenizers can differ 2-5×; this drives `pageindex_max_tokens_per_node` and group sizing — so "20k" is wishful. Big-character-set docs (CJK, formulas) explode in real tokens.
- **`process_none_page_numbers` doesn't propagate freshly-resolved indices.** Its docstring even admits it: "we don't propagate freshly-resolved indices mid-pass" ([toc/page_mapping.py:256-259](src/rag_pageindex/pageindex/toc/page_mapping.py#L256-L259)). Each item is resolved independently against its *original* neighbour window — fine for parallelism, but causes drift when several adjacent items were all unresolved.
- **`process_large_node_recursively` doubles cost on long appendices.** Triggers when `page_span > 10 AND tokens >= max_tokens_per_node`. Hits "Appendix A: tables 1-50" hardest — exactly the kind of node where a flat list of pages is the right answer, not invented sub-structure.
- **Printed TOCs rarely list sub-sub-sections.** This is a *structural* precision limit of the current pipeline, not a bug: Paths A and B build the tree directly from the TOC text, so anything the author didn't print (depth ≥3 headings, sub-sub-sections, intra-chapter breaks) is **never discovered**. `process_large_node_recursively` is the only escape hatch and only fires on `page_span > 10 AND tokens ≥ 4096`, so most missing structure stays missing. This is the strongest argument for the partition reframing in §3.1.

### 2.4 Token-spend hot-spots

| Symptom | Where | Suggested cap |
|---|---|---|
| Path C re-sends full prior JSON every group | [toc/page_mapping.py:425-428](src/rag_pageindex/pageindex/toc/page_mapping.py#L425-L428) | Pass *last N headings only* instead of full structure. |
| `verify_toc` over all items | [tree/builder.py:289](src/rag_pageindex/pageindex/tree/builder.py#L289) | Sample √K (or min 8, max 30); only escalate to full-coverage when sample accuracy ∈ (0.6, 0.95). |
| Summaries: 1 call per node | [tree/summaries.py:82-121](src/rag_pageindex/pageindex/tree/summaries.py#L82-L121) | Batch: K leaves with similar depth in one structured call (`{"summaries":[{node_id,summary}…]}`). |
| `add_node_text` + summary pass keeps full text in memory | [pipeline.py:101-105](src/rag_pageindex/pageindex/pipeline.py#L101-L105) | Stream-then-discard per leaf; never hold full doc text in tree. |
| VLM fallback renders+sends *every* failing page at 144 DPI | [toc/verification_vlm.py:53-153](src/rag_pageindex/pageindex/toc/verification_vlm.py#L53-L153) | Try 96 DPI first; escalate only when the first VLM pass returns `confidence=low`. |

---

## 3. Proposed improvements

### 3.1 Reframe `detect_toc` as a partition hint, not a tree

> **Note:** this section assumes we stay on the current "raw-text + LLM-driven structure inference" architecture. If a markdown intermediate is on the table, see §7 — that approach makes most of this section moot by eliminating the inference step entirely. Treat §3.1 as the best available refactor *within* the existing architecture, and §7 as the architecture-level alternative.

The single biggest structural change *inside the current architecture*. **The current code treats the printed TOC as the authoritative tree** — Paths A and B build the output tree directly from the TOC text, so any heading the author didn't print (almost always: sub-sub-sections, intra-chapter breaks, appendix sub-tables) is silently dropped. The cascade and `process_large_node_recursively` are scaffolding that tries to work around this, badly.

**Proposed model:** demote the TOC from "tree builder" to "cheap page-range partitioner", then *always* run intra-range heading discovery to fill in deep structure.

```
toc_found?  ──yes──►  use it to anchor top-level chunks (1 LLM call for offset)
            ──no───►  whole-doc is one chunk

for each chunk in parallel:
    extract_headings_in_range(chunk_pages)        # current process_no_toc, scoped

verify(sampled across all discovered headings)
fix(incorrect)  →  text fixer, then VLM fixer
```

This collapses the four current code paths into one:

| Today | Becomes |
|---|---|
| Path A — `process_toc_with_page_numbers` | TOC defines chunks; `extract_headings_in_range` fills each one |
| Path B — `process_toc_no_page_numbers` | Resolve page offset first, then same as Path A |
| Path C — `process_no_toc` | One chunk = the whole doc; same `extract_headings_in_range` |
| `process_large_node_recursively` | A leaf that's still too big → re-chunk and re-extract (same function) |

**Wins:**
- **Strictly more precise on TOC-having docs.** Depth ≥3 finally gets discovered instead of being capped at whatever the printed TOC lists.
- **Cheaper than today.** Each chunk's heading-extraction prompt is small (one chapter's worth of pages), and chunks run in parallel. Path C's "growing JSON across groups" pattern (which today re-sends every prior heading) disappears.
- **One function, one prompt** for heading extraction. The three near-duplicate prompts (`generate_toc_init`, `generate_toc_continue`, `add_page_number_to_toc`, `toc_index_extractor`) collapse into one.
- **The cascade goes away.** No fall-through, no thrown-away work — there is only one path.

**Why keep `detect_toc` at all under this model:**
1. It defines free, author-blessed chunk boundaries — much better signal than LLM-invented depth-1 splits.
2. With page numbers, the offset trick (`toc_index_extractor` + `calculate_page_offset`) anchors all top-level boundaries in ~1 LLM call. Without a TOC, you'd have to ask the LLM to locate every top-level heading from scratch.
3. The verifier has a small, named target set ("did these N section starts land on the right pages?") instead of having to score the LLM's free-form structure invention.

**What `detect_toc` is no longer asked to do:** be the tree. The TOC text is consulted *once* to derive chunk boundaries; after that, the heading extractor owns the structure.

### 3.2 Up-front document-type classification

Add a cheap classifier *before* `check_toc`. One LLM call (or even a regex/heuristic) on the first 1-2 pages classifies the doc into:

- `book` / `report` (multi-chapter, long-form prose) → full pipeline
- `paper` (academic) → skip TOC detect, run a lightweight section extractor (Path-C-lite with smaller groups)
- `slides` → "one node per slide" pipeline, no hierarchy invention
- `tabular` / `single-page` → minimal pipeline: doc-level summary only, no tree
- `scanned` (avg tokens/page < threshold) → route to VLM directly

Each branch is a *simple* function. The complexity moves to the dispatcher, but each branch becomes a few dozen lines instead of the current 600-line builder.

Cheap signal that does **not** need a new LLM call:
- Tokens-per-page distribution (slides ~50-200, prose ~500-2000, scans ~0-20).
- Page count.
- Presence of `<physical_index_X>` aspect ratio (slides are landscape) — already available via PyMuPDF.

### 3.3 Tree-precision wins

- **Repair gappy structures in `list_to_tree`** ([tree/builder.py:29-77](src/rag_pageindex/pageindex/tree/builder.py#L29-L77)): when a parent slot is missing, synthesize an empty parent (or re-attach to the nearest existing ancestor). Stops `1.2` from becoming a root.
- **Replace `appear_start` end-index logic with a direct range query.** Ask the LLM (or just use the *next item's start − 1*) and verify by re-checking the boundary page. The "no/yes" appear_start path doubles the LLM cost for marginal benefit.
- **Drop blind `add_preface_if_needed`.** Either omit pre-first-section pages, or extract their actual title from the first 1-2 pages (one extra call, accurate label).
- **Deterministic verify sampling.** Seed `random` with the document hash so cascades reproduce.
- **Use a real tokenizer for token counts.** `tiktoken` (cl100k_base) is "good enough" for most non-OpenAI models too; alternatively cache `count_tokens` per page once at read time using the actual model's tokenizer endpoint (most OpenAI-compatible providers expose `/tokenize`).
- **Propagate freshly-resolved indices in `process_none_page_numbers`.** Two-pass: first parallel pass with original neighbours; second parallel pass with updated neighbours for any still-unresolved items.

### 3.4 Robustness on non-book docs

- **Slides path**: each page becomes a node; titles come from a single batched VLM call ("identify the slide title for each of these N images"). One call per ~10 slides at 96 DPI.
- **Table-heavy path**: detect when a page is ≥80% non-prose (low text density, or many digits-and-pipes per line). For these pages, store the page as a leaf with `text` and skip summary generation (or use a table-specific prompt: "summarize this table"). PyMuPDF can return tables structurally via `page.find_tables()`.
- **Scanned-doc auto-route**: if mean tokens/page < 50 across the first 5 pages, switch directly into VLM mode rather than text-first.
- **Heuristic fast-path for TOC detection**: a regex looking for "Contents" / "Table of Contents" anchored at line start, plus dot-leader patterns (`\.{5,}`), captures ~95% of TOCs at zero LLM cost. Use the existing prompt only on the remaining 5%.

### 3.5 Token efficiency

- **`verify_toc` sample size = √K (clamped 8-30)** with stratified sampling (first / middle / last thirds) instead of full coverage. Drops K-cost verification to O(√K).
- **Batch summaries by level.** [tree/summaries.py:82-121](src/rag_pageindex/pageindex/tree/summaries.py#L82-L121) — pack ≤8 leaf-node texts per prompt, ask for a JSON array of `{node_id, summary}`. Reduces 100-node tree from 100 calls to ~12-15.
- **Drop the `complete`-then-`check_if_toc_transformation_is_complete` continuation loop** ([toc/detection.py:202-227](src/rag_pageindex/pageindex/toc/detection.py#L202-L227)) — it's currently unused (no caller invokes `extract_toc_content`) but should be deleted to reduce surface area.
- **Path-C "continue" should carry only the last N headings**, not the entire growing JSON.
- **Cache LLM responses by `(prompt_hash, model)`.** Re-indexing the same PDF after a config tweak is currently a fresh full run; a local on-disk cache (sqlite or `diskcache`) is cheap and saves 100% of cost during dev iteration.

### 3.6 Code-shape simplifications

- **Collapse `toc_transformer`, `add_page_number_to_toc`, `toc_index_extractor`** — three slightly-different "tag this JSON with physical_index" calls — into one schema-driven helper, parameterised by which inputs are present.
- **Delete the unused dual PDF parser path.** [pdf/reader.py:54-76](src/rag_pageindex/pageindex/pdf/reader.py#L54-L76) supports both PyPDF2 and PyMuPDF, but only PyPDF2 is wired into the pipeline ([pipeline.py:89](src/rag_pageindex/pageindex/pipeline.py#L89)). PyMuPDF has consistently better extraction *and* gives us page images + tables for free — picking it as the single backend removes a branch and unlocks the slide/table paths above.
- **Make `IndexResult.structure` use `TreeNode`** (it's already defined in [tree/types.py:28-40](src/rag_pageindex/pageindex/tree/types.py#L28-L40)) instead of `list[dict[str, Any]]`. The pydantic type already exists but is unused.
- **Remove the `extract_toc_content` / `check_if_toc_extraction_is_complete` / `check_if_toc_transformation_is_complete` dead code path** in [toc/detection.py:110-227](src/rag_pageindex/pageindex/toc/detection.py#L110-L227). These functions are never called from `tree_parser` or `meta_processor`.

---

## 4. Suggested rollout order

Ordered by ratio (impact / risk-of-regression):

| # | Change | Rough impact | Risk |
|---|---|---|---|
| 1 | Cap `verify_toc` to √K stratified sample | ~30-50 % fewer calls on long docs | low — current full-sample is already overkill |
| 2 | Switch default PDF parser to PyMuPDF, delete PyPDF2 branch | better text on multi-column / tables | low |
| 3 | Document-type classifier + `slides` / `tabular` / `scanned` short-circuits | huge for non-book docs | medium — new dispatcher |
| 4 | **TOC-as-partition refactor (§3.1)** — collapse Paths A/B/C + `process_large_node_recursively` into one `extract_headings_in_range` loop | major precision + cost win, *unlocks deep structure* | medium-high — central refactor |
| 5 | Batch summaries by depth level | 5-10× fewer summary calls | low |
| 6 | Real tokenizer for `count_tokens` | safer group sizing on edge docs | low |
| 7 | Repair gappy structures in `list_to_tree` | precision win on small-model output | low |
| 8 | Local response cache keyed by `(prompt_hash, model)` | dev-loop speedup, not prod | low |
| 9 | Delete dead extraction-continuation path | code reduction | trivial |
| 10 | Heuristic TOC pre-detector (regex) | ~4 LLM calls saved per doc | trivial |

Changes 1, 5, 6, 7, 9, 10 are largely independent and can ship as small PRs. **Change 4 is the centrepiece** — it consolidates `meta_processor` + `process_large_node_recursively` + the Path A/B/C cascade into one function, and it is the only one of these changes that actually improves *tree precision* beyond what the printed TOC contains. Everything else is cost or robustness.

---

## 5. Open questions to confirm with the team

1. **Target latency** per document — is the user happy with current 30-90s/doc, or do we need sub-10s? This determines whether batching summaries / caching is worth the engineering vs. just throwing money at a bigger model.
2. **Document mix.** What fraction of indexed docs are books vs. papers vs. slides vs. finance? If slides+finance > 30 %, change #3 (classifier) jumps to #1.
3. **Is VLM fallback actually exercising?** The 0.6 threshold means borderline cases just return best-effort. Have we measured the (accuracy, cost) curve at different thresholds, or is 0.6 a guess?
4. **Do we want hierarchy invention for flat docs at all?** For slides/tables, flat `pages → leaves` may be a better RAG signal than a fabricated tree.
5. **Is reproducibility a requirement?** If so, the random sampling, dict-order dependence, and missing seeds should be addressed first.

---

## 6. Quick reference — files to touch per change

| Change | Files |
|---|---|
| Sample-based verify | [toc/verification.py](src/rag_pageindex/pageindex/toc/verification.py), [tree/builder.py](src/rag_pageindex/pageindex/tree/builder.py) |
| PyMuPDF default + delete PyPDF2 path | [pdf/reader.py](src/rag_pageindex/pageindex/pdf/reader.py), [pipeline.py](src/rag_pageindex/pageindex/pipeline.py) |
| Doc-type classifier | new `pageindex/classifier.py`, dispatcher in [pipeline.py](src/rag_pageindex/pageindex/pipeline.py) |
| TOC-as-partition refactor (§3.1) | [tree/builder.py](src/rag_pageindex/pageindex/tree/builder.py) (replace `meta_processor` + `process_large_node_recursively`), [toc/page_mapping.py](src/rag_pageindex/pageindex/toc/page_mapping.py) (Paths A/B/C collapse into one `extract_headings_in_range`), unify [toc/parsing.py](src/rag_pageindex/pageindex/toc/parsing.py) prompts |
| Batched summaries | [tree/summaries.py](src/rag_pageindex/pageindex/tree/summaries.py), new `generate_node_summary_batch.j2` |
| Real tokenizer | [llm/openai_compatible_client.py](src/rag_pageindex/pageindex/llm/openai_compatible_client.py) (`count_tokens`) |
| Gappy-structure repair | [tree/builder.py:29-77](src/rag_pageindex/pageindex/tree/builder.py#L29-L77) (`list_to_tree`) |
| Response cache | new `pageindex/llm/cache.py`, wrap `OpenAICompatibleClient` |
| Delete dead paths | [toc/detection.py:110-227](src/rag_pageindex/pageindex/toc/detection.py#L110-L227) |
| Regex TOC pre-detector | [toc/detection.py](src/rag_pageindex/pageindex/toc/detection.py) (`find_toc_pages`) |

---

## 7. Alternative architecture: markdown intermediate

The proposals in §§2-6 all live inside the current premise: *raw text comes in, an LLM infers structure*. There is a more radical alternative — **convert the PDF to markdown first, then read the tree out of the markdown.** Indexing is one-time per document in a RAG system, so a slower conversion stage is acceptable in exchange for token cost and precision wins.

### 7.1 The core insight

Today the LLM is asked to reason about heading hierarchy because **the text extractor threw that information away**. PyPDF2 / PyMuPDF's `extract_text()` returns flat text — font size, position, weight, the visual signals that make a heading a heading, are gone. The pipeline then spends 200-600 LLM calls reconstructing what was visually obvious in the source.

If the conversion preserves structure as markdown (`# Chapter`, `## Section`, `### Subsection`, tables, lists, math), then **tree extraction collapses to a regex over `^#{N} `**. The entire `toc/` subsystem becomes dead code.

### 7.2 What it gains

- **Tree extraction becomes structural, not inferential.** A markdown walker (~50 LOC) replaces `meta_processor` + `process_large_node_recursively` + the A/B/C cascade. You cannot "miss" a sub-sub-section the way Paths A/B do today — if the converter emitted `### 2.3.1 Foo`, the tree has it. **This is a larger precision win than the §3.1 refactor.**
- **Tables, math, lists, slides preserved verbatim.** Finance docs benefit most: a markdown table is something the retrieval agent can reason over, where today PyPDF2 returns digit-and-space soup.
- **The cascade disappears, and with it ~80% of the *structure-extraction* prompts.** Of the 21 templates in [pageindex/prompts/](src/rag_pageindex/pageindex/prompts/), about 15 exist purely to compensate for lost structure: `check_toc`, `detect_page_index`, `toc_to_json`, `toc_to_json_continue`, `extract_toc_content`, `extract_toc_continue`, `check_toc_extraction_complete`, `check_toc_transformation_complete`, `generate_toc_init`, `generate_toc_continue`, `generate_toc_vlm_range`, `toc_index_extractor`, `add_page_number_to_toc`, `single_toc_item_index_fixer`, `single_toc_item_index_fixer_vlm`. After the refactor only `check_title_appearance` (for the verifier), `generate_node_summary*`, and `generate_doc_description` survive.
- **Important caveat — summaries are *not* optional, they are the retrieval signal.** PageIndex is a vectorless RAG: the agent picks a route by *reading the tree*. Titles alone ("Methods", "Results", "Appendix A") are far too thin a signal — the per-node `summary` is what makes routing decisions correct. So the markdown architecture removes the **structure-inference** LLM cost (200-500 calls) but **not** the **summary-generation** cost (~one per node, batched). Summaries are load-bearing under any architecture; this is not a cost the markdown path saves.
- **Unified path for scanned and text-native PDFs.** Both go through the converter; there is no separate "text path with VLM fallback" dichotomy. Today's text-vs-vision branching disappears.
- **`appear_start`, `add_preface_if_needed`, random-sampled `verify_toc`, gappy-structure repair in `list_to_tree`** — all of them lose their reason to exist.

### 7.3 What it trades

- **A new silent-failure mode.** Today a bad text extraction produces verifiable damage (`verify_toc` accuracy drops, the cascade reacts). A bad markdown conversion produces a *plausible-looking* tree that is quietly wrong — invented heading, wrong depth, dropped section. Mitigation: keep a thin verifier that samples K headings, renders the corresponding PDF page, and asks a VLM "does this heading exist on this page at this level?". Same shape as today's `verify_toc`, ~√K calls, but with a much higher signal-to-noise ratio because the question is binary.
- **Small generative LLMs hallucinate structure.** A 3-12B instruction-tuned model converting a page to markdown will invent h2/h3 levels based on prose feel, drop "boring" headings, and normalise capitalisation. **The right answer here is to *not* use a generative LLM for the conversion** — use a deterministic layout-aware pipeline instead.
- **Page-mapping must survive the conversion.** Retrieval still has to return PDF page numbers. The converter must inject `<page_X>` markers (HTML comments are markdown-safe) at every page boundary, and the tree walker carries them into `start_index` / `end_index`. Most production tools below support this out of the box.
- **The "reasoning-based RAG" framing shifts.** The intelligence now lives in the converter, not in PageIndex. PageIndex becomes a thin structural pass. Arguably a more honest separation of concerns, but it is a project-identity change.

### 7.4 Choice of converter

Two families:

**Family A — generative LLM (small model, e.g. gemma-4-3b, phi-4-mini).** Flexible, no extra deps beyond the existing LLM client. Hallucination is the dominant risk. Best used only when the structure is so loose that deterministic layout analysis fails (free-form notes, handwritten scans).

**Family B — deterministic layout-aware pipelines (recommended).** These are vision + OCR + layout-analysis tools that emit markdown without generation. No hallucination, reproducible output, mostly CPU-runnable.

| Tool | Strengths | Notes |
|---|---|---|
| **[Marker](https://github.com/VikParuchuri/marker)** | books, papers, very good table support, math via LaTeX | mature, MIT, the default recommendation |
| **[Docling](https://github.com/DS4SD/docling)** (IBM) | production-grade, strong table-structure reconstruction | MIT, growing fast |
| **[MinerU](https://github.com/opendatalab/MinerU)** | Chinese-language docs, scientific content with formulas | Apache-2 |
| **[Nougat](https://github.com/facebookresearch/nougat)** | academic papers specifically | needs GPU for reasonable throughput |
| **Microsoft markitdown** | office docs (slides, docx), light-weight | not ideal for complex PDFs but excellent for ppt/docx |

For a mixed corpus (books, reports, slides, finance), Marker or Docling as the default with a small-LLM fallback for unusual inputs is the strongest baseline.

### 7.5 Proposed architecture

```
PDF ──► [Marker / Docling]  ──►  markdown with <page_X> markers
        └── deterministic, layout+OCR, no LLM

markdown ──► [tree_from_markdown.py]  ──►  tree(node_id, start_page, end_page)
             └── pure AST/regex walker, no LLM

(optional) tree ──► [verify_sample_with_vlm]  ──►  flag suspect headings
                    └── ~√K VLM yes/no calls

tree ──► [batched summaries (per §3.5)]  ──►  IndexResult
        └── REQUIRED — load-bearing for vectorless retrieval, not optional
```

### 7.5.1 Honest cost comparison

| Cost component | Today | Markdown architecture |
|---|---:|---:|
| Structure detection + extraction + verify + fix | **200-500 LLM calls** | 0 LLM calls (deterministic converter) + ~√K optional VLM verifier calls |
| Per-node summaries (load-bearing for routing) | ~2K LLM calls, one per node | same node count, but batched per §3.5 → ~K/8 calls |
| Doc description | 1 | 1 |
| PDF→markdown conversion | n/a | 1× per document (CPU, no LLM) — adds wall-clock, no token cost |

The headline win is the **first row** (structure-inference cost goes to zero), plus the structural-precision gain that comes from not throwing heading information away in the first place. The summary cost is unchanged in shape but should be batched (§3.5) in either architecture — that is an orthogonal lever, not part of the markdown trade.

The structure is *also more complete* — depth ≥3 is finally captured — which means the tree the agent reads has more, and better-described, decision points.

### 7.5.2 Summaries are the retrieval signal — implications

Because routing accuracy depends on summary quality, the markdown architecture should treat summary generation as the *primary* LLM workload, not an afterthought:

- **Pick a stronger model for summaries than for structure.** Today's pipeline forces one model to do both. Once structure is deterministic, you can run a small/cheap model nowhere, and a stronger model only on the ~K/8 batched summary calls — net spend usually drops.
- **Summary prompts should be aware of position in the tree.** The current `generate_node_summary.j2` just sees the node's text. A better prompt sees the path (parent titles) and sibling titles, so the summary disambiguates the node from peers — exactly what an agent needs to route between siblings.
- **Failed summaries are now silent retrieval failures.** Today a missing summary degrades routing; you should keep the existing logger.warning, but also fail the index build (or flag the node) rather than write `summary=""`. Empty summaries are worse than missing nodes because the agent will route to them and get nothing.

### 7.6 Migration shape

This is not a one-PR change. A staged rollout:

1. **Add an optional converter stage.** New module [pageindex/markdown/](src/rag_pageindex/pageindex/markdown/) with a `MarkdownConverter` protocol and a default Marker implementation. Off by default, behind `pageindex_input_format: pdf | markdown`.
2. **Implement `tree_from_markdown`** as a parallel path to `tree_parser`. When `input_format=markdown`, skip the entire `toc/` subsystem.
3. **Add the VLM verifier** as an optional pass.
4. **Run both paths in shadow** on the test corpus, compare trees, measure precision and depth coverage.
5. **Flip the default** once shadow runs are stable, deprecate the legacy text-inference path, then delete `toc/detection.py`, `toc/parsing.py`, `toc/page_mapping.py`, `toc/verification*.py`, and the bulk of `toc/parsing_vlm.py` / `toc/verification_vlm.py`.

After step 5, the entire `pageindex/` package is roughly: `markdown/` (in) + `tree/` (build + summarise) + `llm/` + `pipeline.py`. The `toc/` directory disappears. Expect a ~60% LOC reduction.

### 7.7 When to *not* take this path

- If the corpus is heavily handwritten / non-standard layouts where deterministic converters underperform. (Then either stick with §3.1, or use a generative converter with the VLM verifier as a tighter loop.)
- If pulling in Marker/Docling as a dependency is unacceptable (size, license, deployment).
- If the project's identity as "LLM-built routing tree" matters more than precision.

Outside those cases: the markdown intermediate is the structurally cleaner answer, and the §§2-6 work becomes incremental polish rather than a path forward.

### 7.8 Relationship to §§3-6

| Section | Status under markdown architecture |
|---|---|
| §3.1 TOC-as-partition | Obsolete — markdown headings *are* the partition |
| §3.2 Doc-type classifier | Still useful — slides/scans benefit from converter selection |
| §3.3 Tree-precision wins | Mostly obsolete — `appear_start`, `list_to_tree` gappy repair, preface logic all disappear |
| §3.4 Non-book robustness | Subsumed — converter handles slides/tables/scans natively |
| §3.5 Token efficiency | Summaries-batching becomes **more important**, not less — it is now the dominant LLM-cost lever and the load-bearing retrieval signal (§7.5.2) |
| §3.6 Code-shape simplifications | Subsumed by the ~60% deletion described above |

---

## 8. Alternative architecture: VLM-per-page (or per-page-batch)

A third option: skip both text inference (§§1-6) and the deterministic converter (§7), and just **send each page as an image to a VLM, asking for the heading(s) + a description of the page in one call.** Aggregate the responses into a tree.

The intuition is right and the cost objection is partly misleading. Worked through honestly:

### 8.1 The non-obvious token accounting

Per-page image tokens at 144 DPI sit around **1.5k-3k tokens** depending on provider (Anthropic / OpenAI high-detail tiling, OpenRouter passthrough). Per-page text tokens for typical prose are 300-1500. So per call, an image is ~2-5× more expensive than the same page as text.

**But today's text pipeline sends each page in several different prompts:**

| Stage | Pages each one sees |
|---|---|
| `toc_detector_single_page` | first 4 pages |
| `toc_index_extractor` | first ~K pages of body (for offset calibration) |
| `process_none_page_numbers` | window of pages around each unresolved item |
| `process_no_toc` (Path C) | every page, often re-sent in "continue" prompts |
| `verify_toc` | one page per sampled item (full-sample by default) |
| `fix_incorrect_toc` | a page range per failing item |
| `check_title_appearance_in_start_concurrent` | one page per item |
| `generate_node_summary` | every page (concatenated into leaf `text`) |

A given page text typically appears in **4-8 different prompts**. Net text tokens per page in today's pipeline ≈ 8-15 k. A VLM-per-page approach sends the page **once** at ~2 k image tokens.

**So the cost gap is closer than it looks, and on text-poor or scan-heavy documents the VLM path is actually cheaper.** The "image costs more" intuition is per-occurrence, not per-document.

### 8.2 What the VLM-per-page architecture looks like

```
PDF ──► render pages (96-144 DPI)
     ──► for each batch of N pages (parallel):
           VLM call: "for each page, return:
             - heading(s) starting on this page with depth level
             - a 1-2 sentence description of what this page covers"
     ──► aggregate responses into a flat list of (page, headings, description)
     ──► fold headings into a tree (depth levels → parent/child)
     ──► descriptions become node `summary` field directly,
         either per-page (for leaves) or rolled up bottom-up (for parents)
```

Key shape difference from §7: **structure and summary come from the same VLM call.** The current pipeline runs structure (~K calls) and summary (~K calls) as two separate passes. Fusing them halves the call count.

For a 200-page doc, batching 8 pages per call: **~25 LLM calls total**. Compared to today's 250-600, that's a 10-20× reduction in call count, with each call being a few-image VLM call instead of a text-only one. Net tokens: similar or lower than today on most documents; meaningfully lower on table-heavy / scan-heavy ones.

### 8.3 What this gains over both alternatives

- **Vs. today's pipeline**: 10-20× fewer LLM calls, depth ≥3 finally visible, native handling of tables/charts/scans/slides, no cascade.
- **Vs. §7 (markdown intermediate)**: no extra dependency (Marker/Docling), no need for a `<page_X>` injection step (the VLM already knows which page it's on because you tell it), and **structure + summary come for free in one call**.
- **Unified path for every document type.** The §7 "converter selection" subtlety (Marker for books, markitdown for slides, Nougat for papers) collapses — one prompt, one model, all doc types.

### 8.4 What this trades

- **Hallucination risk per page is higher than for a deterministic converter.** A VLM can invent a heading from a centred bold line that is actually a figure caption. Mitigation: a second sampled-VLM verifier ("is this string a heading on this page?") — same shape as §7's verifier, ~√K calls.
- **Cross-page section continuity needs care.** A section that starts on page 5 and continues to page 9 produces 5 page-level descriptions; you must roll them up into one section-level summary, not concatenate. This is just the existing bottom-up reducer in [tree/summaries.py:_generate_parent_summary](src/rag_pageindex/pageindex/tree/summaries.py#L50), applied at "section gathers pages" granularity.
- **Heading-depth detection is harder than it sounds.** A VLM seeing a single page can't reliably tell whether a bold line is `##` or `###` — depth is *relative* to other headings in the document. Mitigation: a second short pass after collection that normalises depth levels using font-size hints from the original PDF (PyMuPDF exposes per-span font sizes), or simply: ask the VLM to return a sortable visual rank (font size class small/medium/large) and assign depth post-hoc.
- **Small VLMs (≤12B) miss small headings in dense pages.** This is a real failure mode on academic papers with 10-pt section headers. Either use a stronger VLM, or push the DPI up to 192 for dense pages.
- **Image-token cost dominates per call.** Cheap text models cannot be used for this; you're paying VLM rates throughout. On a fully text-native corpus with clean TOCs, §7 is still cheaper because Marker/Docling needs zero LLM tokens for structure.
- **Page-mapping is trivial here**, unlike §7. Each VLM call carries explicit page indices in the prompt; responses can be tied back to pages by construction.

### 8.5 Where this clearly wins

- **Scanned / image-only PDFs** — VLM is the only viable path; today's pipeline produces a degenerate tree.
- **Slide decks** — one page = one slide, structure is visually obvious, descriptions capture the slide's content directly.
- **Financial reports, earnings decks, dashboards** — tables and charts are interpretable in images, opaque in text extraction.
- **Mixed / unknown corpus** — one architecture handles everything, no per-doc classifier needed.

### 8.6 Where this loses

- **Long text-native books with clean TOCs.** The deterministic converter in §7 is strictly cheaper (no per-page LLM cost) and equally precise. VLM-per-page is paying for capability you don't need.
- **Math-heavy academic papers.** Nougat-style specialised models reconstruct LaTeX more faithfully than a general VLM; small VLMs often garble equations.
- **Cost-sensitive batch indexing** of large corpora where most docs are clean prose. The constant per-page VLM cost adds up.

### 8.7 Combined architecture (the pragmatic answer)

The three options are not exclusive. A realistic production system probably looks like:

```
PDF ──► quick classify (page count, tokens/page, has-images, aspect ratio)
        ├── text-native + clean TOC  ──► §7 markdown intermediate (Marker)
        ├── text-native + messy TOC  ──► §7 markdown intermediate (Marker)
        ├── slides / mixed layout    ──► §8 VLM-per-page (this section)
        ├── scanned (≈0 tokens/page) ──► §8 VLM-per-page (this section)
        └── single-page / one-table  ──► trivial flat-leaf, no tree extraction
```

Each branch is small. The dispatcher itself is ~20 lines of heuristics. The §3.5 batched-summary path applies to all branches.

### 8.8 Three-way comparison

| Dimension | Today (text + LLM cascade) | §7 markdown intermediate | §8 VLM-per-page |
|---|---|---|---|
| LLM calls per 200-page doc | 250-600 | ~10-25 (summaries only) | ~25-40 (structure + summary fused) |
| Tokens per doc | High (each page in 4-8 prompts) | Lowest (text-only summaries) | Mid (image tokens, but each page seen once) |
| Depth ≥3 coverage | Poor (TOC-bounded) | Excellent (every `###` captured) | Good (every visible heading captured) |
| Scan / image-only PDFs | Broken | Depends on converter (Marker handles via OCR) | Native — best path |
| Slides | Poor (invented hierarchy) | Good with markitdown | Native — best path |
| Tables, charts | Poor (digit soup) | Good (markdown tables) | Excellent (visual reasoning) |
| Math-heavy papers | Poor | Excellent with Nougat | Mid (small VLMs garble LaTeX) |
| Reproducibility | Low (random sampling, LLM nondeterminism) | High (deterministic converter) | Mid (VLM nondeterminism, low at T=0) |
| New dependency | none | Marker / Docling | none (uses existing `LLMClient`) |
| Hallucination risk | High (everything is LLM-inferred) | Low (deterministic converter; LLM only for summaries) | Mid (one VLM pass per batch) |
| Best for | nothing, honestly | clean text-native books, papers | scans, slides, finance, mixed corpora |

The honest reading: **§7 wins on clean books/papers, §8 wins on everything else, today's pipeline wins on nothing.** A combined architecture (§8.7) gets the strengths of both.
