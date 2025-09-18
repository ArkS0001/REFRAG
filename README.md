# REFRAG

RAG has gotten a new update, and it's called REFRAG (Rethinking RAG-based Decoding) by Meta Superintelligence Labs!
I work extensively with retrieval-augmented generation (RAG) and often hit these limits:
• Latency: time-to-first-token (TTFT) grows quadratically with context length
• Memory: key-value cache grows linearly
• Inefficiency: many retrieved passages are only loosely relevant, wasting compute
REFRAG changes the game. Instead of handing the LLM a giant concatenation of mostly irrelevant text, it feeds dense chunk embeddings and only expands details when needed.

Highlight of the Paper: Selective Chunk Expansion

A lightweight reinforcement-learning (RL) policy scores each chunk’s importance. Chunks predicted to be critical stay in full token form, while the rest stay compressed as embeddings. This lets the model zoom in on key passages during decoding without paying the cost of processing every token.

Key Shift
Standard RAG: uses embeddings only to search, then sends full tokens to the LLM.
REFRAG: uses embeddings to search and decode, replacing most raw tokens with pre-computed chunk embeddings and selectively expanding critical chunks on-the-fly.

Result: dramatically lower latency, smaller memory footprint, and support for far longer contexts.

Intuitively, we can: 
Think of standard RAG as:
“Use embeddings to find the books, then hand the whole books to the LLM to read.”
REFRAG says:
“Use embeddings to find the books and hand the LLM a summary vector of each book, only opening full pages when absolutely necessary.”

This move from “embeddings just to search” to “embeddings to search and to decode” is the core difference that makes REFRAG a breakthrough for long-context and retrieval-heavy LLM systems.
