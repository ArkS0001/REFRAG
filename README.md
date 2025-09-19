# REFRAG

<img width="1087" height="793" alt="725f0222-0b11-4d10-8215-255525e7887f" src="https://github.com/user-attachments/assets/520ad94a-8fb6-4ac6-9b3a-40ef0d357051" />
<img width="1108" height="793" alt="9226edb3-51b9-482c-84d6-3bd68e35a881" src="https://github.com/user-attachments/assets/7ae707ce-006f-4ba2-b0dd-3fbfe0d4dcf4" />

<img width="1066" height="793" alt="44743e5e-67a9-426c-86fc-866a2ab90ef1" src="https://github.com/user-attachments/assets/8b3ee4c8-f4b7-4756-aeba-c0f68cac0dd2" />




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

 *\~30% latency reduction* depends on:

1. **Your dataset size & chunking**

   * In your current setup, you’re multiplying the same sentence 30×, so the chunks are very repetitive.
   * That makes retrieval + expansion trivial → REFRAG can’t really “skip work,” so speed gains might not be obvious.

2. **LLM backend**

   * You’re using **GPT-2** (small, local model). Its generation latency is *tiny compared to embedding/retrieval*, so even if REFRAG expands fewer chunks, the total time difference is small.
   * The **30% claim** usually comes from large LLMs (like GPT-3.5, LLaMA-70B) where prompt length dominates cost.

3. **Expansion policy**

   * Your code expands up to `max_expands=min(8, k)`. That already limits prompt bloat.
   * If you raise `k` (say 64 or 128), **standard RAG prompt explodes**, but REFRAG still stops around \~8 expansions → then you’ll see a big gap.

---

⚡ In short:

* Yes, your code is structured so REFRAG *can* beat RAG.
* But with **small GPT-2 + repetitive docs**, the benchmark won’t show a strong 30% speedup.
* If you swap in a bigger model (even `facebook/opt-1.3b`) and set `k=64+`, you should see REFRAG latency curve flatten while RAG keeps rising.

---

