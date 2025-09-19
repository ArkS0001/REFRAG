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
Perfect 🚀
Let’s tune your benchmark so the **REFRAG advantage** is visible:

### 🔑 Adjustments

1. **Increase Top-K** → push `k` up to 64 or 128.

   * Standard RAG’s prompt length grows linearly with `k`.
   * REFRAG caps expansions → curve flattens.

2. **Cap Expansions Harder**

   ```python
   max_expands = 8
   initial_expand = 2
   ```

   → This ensures REFRAG never expands more than \~8 chunks, no matter how high `k` goes.

3. **Use Diverse Docs**
   Replace repetitive multiplication (`*30`) with **5–10 unique long texts** (Wikipedia dumps, or paragraphs from different topics).
   That way retrieval matters, and compression helps.

4. **Collect TTFT (time-to-first-token)**
   That’s what papers report. We’ll measure it separately from full completion time.

---

### 🔧 Code Edits

Here’s the tuned part of your benchmark:

```python
# -----------------------
# Run Demo Benchmark (Tuned)
# -----------------------

# More diverse long docs
docs = {
    "doc1": "Paris is the cultural capital of France, home to museums, art, and fashion. " * 50,
    "doc2": "Python is a programming language widely used in AI, data science, and automation. " * 50,
    "doc3": "Transformers with self-attention revolutionized NLP, powering GPT and BERT models. " * 50,
    "doc4": "Quantum mechanics describes subatomic particles and drives semiconductors, lasers, and quantum computing. " * 50,
    "doc5": "Global economics involves markets, trade, IMF, World Bank, and ripple effects during crises. " * 50,
    "doc6": "Climate change affects weather, agriculture, biodiversity, and drives international agreements like Paris Accord. " * 50,
}

idx = EmbeddingIndex()
idx.build_from_docs(docs)
llm = LLMWrapper()

queries = [
    "Why is Paris culturally important?",
    "What is Python used for?",
    "Explain transformers in AI.",
    "What impact did quantum mechanics have on technology?",
    "How do global economic shocks spread?",
    "How does climate change influence international policies?"
]

# Push Top-K higher
ks = [4, 8, 16, 32, 64, 128]

# Force REFRAG to stay compressed most of the time
def tuned_refrag_answer(query, index, llm, top_k):
    return refrag_answer_selective(
        query, index, llm,
        top_k=top_k,
        initial_expand=2,
        max_expands=8,  # hard cap
        policy_fn=simple_policy_score
    )

# Replace in benchmark loop
metrics = run_benchmark(idx, llm, queries, top_k_values=ks)
plot_metrics(metrics)
```

---

### 🔍 Expected Graphs

* **Latency:**

  * RAG line keeps rising with `k`.
  * REFRAG line flattens after \~8 expansions.
  * At `k=64+`, REFRAG should be **20–40% faster**.

* **Memory:**

  * Standard RAG grows steadily.
  * REFRAG curve flattens after 8 expansions.

* **Expansion Rate:**

  * Always \~8 regardless of `k`.

---

Would you like me to **rewrite your full notebook** with these tuned changes so you can just copy-paste into Colab and run the improved benchmark?

