# Hacker News Intelligence Tool

🎥 **[Demo Video: Watch the tool analyze 'SQLite in production'](https://youtu.be/hgAchcrsWPM)**

An AI-powered intelligence dashboard that semantically translates conversational questions, searches Hacker News, and uses local LLMs to extract technical arguments, pros, and cons from massive community discussions.

## Stages Completed
- Phase 1: The Digest (Structured pros/cons extraction)
- Phase 2: Conversational Chat (Interactive Q&A with the fetched Hacker News context)

## Features
* Semantic Query Optimization: Converts conversational NLP queries into Algolia-optimized strict keywords while preserving core entities.
* Multithreaded Data Ingestion: Uses thread pooling to fetch hundreds of deep-nested Hacker News comments concurrently, cutting network latency by 80%.
* Dynamic Hardware Safeguards: Features a UI-driven context window synchronizer. It automatically packs and truncates RAG contexts to perfectly match your local GPU/RAM limits, completely preventing HTTP 400 Context Overflow errors.
* Fault-Tolerant JSON Extraction: Implements bracket-snatching and key-mapping fallbacks to survive LLM hallucinations, markdown formatting, and conversational filler.
* Interactive RAG Chat: Allows users to have a context-aware conversation directly with the fetched Hacker News thread to dive deeper into specific arguments and user opinions.
* 100% Local Inference: Runs entirely locally via LM Studio, ensuring absolute privacy and zero API costs.

## Prerequisites
1. Python 3.9+
2. LM Studio (with a loaded quantized model, e.g., Llama 3 8B Instruct).

## Getting Started
1. Start the Local LLM Server: Open LM Studio, load your preferred model, start the Local Inference Server (default: http://localhost:1234/v1). Ensure CORS is enabled.
2. Install Dependencies: pip install -r requirements.txt
3. Run the Application: streamlit run app.py

## Usage Nuances & Hardware Sync
Local LLMs are bound by physical hardware constraints (VRAM). 
* Context Limit Slider: In the sidebar, you MUST set the "LM Studio Context Limit" to match the n_ctx setting in your LM Studio hardware configuration. 
* Automatic Truncation: The app will automatically calculate a safe buffer and cleanly pack the top-voted comments into the prompt. If the thread is too large for your hardware, the app will gracefully truncate the bottom comments rather than crashing the server.
