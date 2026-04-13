# Technical Architecture & Engineering Report

## 1. System Architecture
The Hacker News Intelligence Tool is built on a decoupled architecture prioritizing speed, hardware resilience, and strict data formatting.
* Frontend: Streamlit (Reactive UI, dynamic hardware syncing, state management).
* Orchestration & NLP: LangChain (Prompt templating, JSON parsing schemas).
* Data Layer: Algolia Hacker News REST API (Strict keyword matching, thread traversal).
* Inference Engine: LM Studio API (Local OpenAI-compatible endpoint).

## 2. Overcoming Search Paradigm Mismatches
A major engineering hurdle was bridging the gap between Natural Language (user queries) and strict lexical matching (Algolia's search engine).
* The "Goldilocks" Optimizer: When passing raw NLP to Algolia, results fail. If an LLM over-prunes to just stop-words, it leaves verbs that ruin lexical searches. We implemented a semantic translation prompt that converts conversational phrases into industry-standard terms (e.g., "how to make prompts" -> "prompt engineering") while enforcing strict few-shot boundaries to prevent the LLM from hallucinating unrelated technologies (the "Anchoring" problem).

## 3. High-Performance Concurrency
Hacker News threads often contain 300+ comments. Fetching these sequentially created a massive I/O bottleneck (~15+ seconds). 
* Thread Pooling: We implemented concurrent.futures.ThreadPoolExecutor in the API pipeline, allowing parallelized fetching of comment trees. This reduced data ingestion time by over 80%.
* Memoization: Integrated @st.cache_data to cache API responses and LLM optimizations, resulting in instantaneous sub-millisecond loads for repeated queries.

## 4. Hardware Safeguards & Context Window Management
Local LLMs frequently crash when payloads exceed their configured n_ctx limit (Context Window Overflow). 
* Dynamic Hardware Sync: We exposed a dynamic context-limit parameter in the UI. 
* Iterative Context Packing: Rather than blindly slicing text arrays (which halves sentences and destroys LLM attention), the pipeline iteratively packs the highest-voted comments into the payload string. Once it approaches the dynamic hardware limit (minus a 1,500 token generation buffer), it safely breaks the loop. This guarantees 0 crashes while preserving the highest-signal data.

## 5. Resilient JSON Extraction
Quantized 8B models suffer from instruction drift, often wrapping structured JSON outputs in markdown backticks, adding conversational filler, or hallucinating dictionary keys.
* The Bracket Snatcher: Replaced fragile LangChain output parsers with a custom Python extraction layer. The pipeline explicitly locates the first '{' and last '}', ignoring all surrounding markdown or filler.
* Schema Alignment: Implemented a fault-tolerant .get() fallback mapping system to catch known key hallucinations, ensuring the Streamlit UI never encounters a KeyError.
