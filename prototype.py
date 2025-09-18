"""
refrag_prototype.py
Proof-of-concept implementation of the REFRAG idea (selective chunk expansion).
- Chunking + embeddings (sentence-transformers)
- FAISS index for retrieval
- Lightweight policy to select which chunks to expand
- Iterative decoding loop that expands chunks on-demand

Requirements:
pip install sentence-transformers faiss-cpu transformers torch tqdm

Notes:
- This is an application-level prototype. REFRAG's paper uses model-level mechanisms (embedding->soft-token
  mappings, KV injection) for larger speedups, but this prototype implements the "compress, sense, expand" loop
  and an RL-capable policy.
- See paper: REFRAG: Rethinking RAG based Decoding (arXiv). (cited in conversation)
"""

import os
import math
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# -----------------------
# Config
# -----------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # lightweight
LLM_MODEL = "gpt2"  # replace with causal LLM of choice (or use OpenAI API wrapper)
CHUNK_TOKENS = 64
TOP_K = 12           # number of retrieved chunks to consider
INITIAL_EXPAND = 2   # number of chunks to expand before starting decoding
MAX_EXPANSIONS = 6   # upper bound of expansions during decoding
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Utilities: chunker, embedder, index
# -----------------------
def chunk_text(text: str, tokens_per_chunk: int = CHUNK_TOKENS) -> List[str]:
    """Naive whitespace chunker by approximate token count (use tokenizer for precision)."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + tokens_per_chunk]))
        i += tokens_per_chunk
    return chunks


class EmbeddingIndex:
    def __init__(self, embed_model_name=EMBED_MODEL, dim=384):
        self.encoder = SentenceTransformer(embed_model_name)
        self.encoder.max_seq_length = 512
        self.dim = self.encoder.get_sentence_embedding_dimension()
        self.index = None
        self.metadata = []  # list of dicts with 'text', 'doc_id', 'chunk_id'
        self.embeddings = None

    def add_chunks(self, chunk_texts: List[str], doc_id: str):
        embs = self.encoder.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
        if self.index is None:
            self.embeddings = embs
            self.index = faiss.IndexFlatIP(self.dim)
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
        else:
            # append
            prev = self.embeddings
            self.embeddings = np.vstack([prev, embs])
            self.index.reset()
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
        base_id = len(self.metadata)
        for i, txt in enumerate(chunk_texts):
            self.metadata.append({"text": txt, "doc_id": doc_id, "chunk_id": base_id + i})

    def build_from_docs(self, docs: Dict[str, str]):
        """docs: dict doc_id -> text"""
        all_chunks = []
        meta = []
        for doc_id, text in docs.items():
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            for c in chunks:
                meta.append({"text": c, "doc_id": doc_id})
        embs = self.encoder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)
        import numpy as np
        self.embeddings = embs
        self.index = faiss.IndexFlatIP(self.dim)
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        self.metadata = meta

    def query(self, q_text: str, k=TOP_K) -> List[Tuple[float, Dict]]:
        q_emb = self.encoder.encode([q_text], convert_to_numpy=True)
        import numpy as np
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            meta = self.metadata[idx].copy()
            meta['score'] = float(score)
            results.append((float(score), meta))
        return results

# -----------------------
# Lightweight policy network (scores chunk importance)
# -----------------------
class PolicyNet(nn.Module):
    """
    Small MLP that takes feature vector for (query, chunk, decoder_state_proxy) and outputs selection score.
    For prototype: features are concatenated embeddings (query_emb + chunk_emb) and a scalar proxy (entropy).
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, q_emb, chunk_emb, decoder_proxy):
        # q_emb, chunk_emb: tensors (batch, emb_dim)
        # decoder_proxy: scalar or tensor (batch,1)
        x = torch.cat([q_emb, chunk_emb, decoder_proxy.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


# -----------------------
# LLM wrapper (simple)
# -----------------------
class LLMWrapper:
    def __init__(self, model_name=LLM_MODEL, device=DEVICE):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def generate(self, prompt: str, max_new_tokens=128, stream=False):
        # For simplicity: run generate and return text. Streaming hooks can be added for TTFT experiments.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the newly generated portion if you wish:
        return text

    def next_token_entropy_proxy(self, prompt: str):
        # Proxy for decoder uncertainty: get logits for next token and compute entropy.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # (batch=1, vocab)
            probs = torch.softmax(logits, dim=-1)
            ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
            return float(ent.cpu().item())


# -----------------------
# REFRAG-like run: iterative decoding with selective expansion
# -----------------------
def refrag_answer(query: str, index: EmbeddingIndex, llm: LLMWrapper, policy: PolicyNet=None,
                  initial_expand=INITIAL_EXPAND, max_expansions=MAX_EXPANSIONS):
    # 1) retrieve top-K chunks (embedding compression)
    top = index.query(query, k=TOP_K)
    # Represent compressed chunks as short summaries (or placeholders)
    compressed_view = [f"[CHUNK_{i}: emb_score={s:.3f}]" for i, (s, m) in enumerate(top)]

    # 2) expand top-N by score as initial context
    expansions = {}
    for i, (score, meta) in enumerate(top[:initial_expand]):
        expansions[i] = meta['text']

    # helper: build prompt from expansions + compressed placeholders
    def build_prompt(expansions_map):
        expanded_texts = []
        for i, (score, meta) in enumerate(top):
            if i in expansions_map:
                expanded_texts.append(f"CHUNK_{i} (expanded):\n{expansions_map[i]}\n")
            else:
                # place compressed representation (could be a short auto-summary; here we use placeholder)
                expanded_texts.append(f"CHUNK_{i} (compressed): {meta['text'][:80]}... [EMB:{score:.3f}]")
        ctx = "\n\n".join(expanded_texts)
        prompt = f"Use the following retrieved chunks to answer the query.\n\nContext:\n{ctx}\n\nQuery: {query}\n\nAnswer concisely:"
        return prompt

    prompt = build_prompt(expansions)
    # get initial uncertainty proxy
    ent = llm.next_token_entropy_proxy(prompt)

    # iterative decode loop: ask LLM to produce up to N tokens, but allow policy to add expansions if proxy indicates
    generated = ""
    expansions_done = len(expansions)
    while True:
        prompt = build_prompt(expansions) + "\n\nPreviously generated:\n" + generated + "\nContinue:"
        # call LLM for a small chunk (e.g., 32 tokens) to let policy evaluate
        out = llm.generate(prompt, max_new_tokens=32)
        # naive way to extract newly generated text:
        newly = out[len(prompt):] if out.startswith(prompt) else out
        generated += newly
        # compute new proxy
        ent = llm.next_token_entropy_proxy(prompt + newly)
        # Use policy (if provided) to score remaining chunks
        if policy is not None and expansions_done < max_expansions:
            # prepare batch of features
            import numpy as np
            q_emb = torch.tensor(index.encoder.encode([query]), dtype=torch.float32).to(DEVICE)
            # collect chunk embeddings for candidate (not yet expanded) chunks
            candidates = []
            cand_idx = []
            for i, (score, meta) in enumerate(top):
                if i in expansions:
                    continue
                cand_idx.append(i)
                candidates.append(meta['text'])
            if candidates:
                cand_embs = torch.tensor(index.encoder.encode(candidates), dtype=torch.float32).to(DEVICE)
                q_emb_rep = q_emb.repeat(cand_embs.shape[0], 1)  # (n, dim)
                ent_rep = torch.tensor([ent] * cand_embs.shape[0], dtype=torch.float32).to(DEVICE)
                scores = policy(q_emb_rep, cand_embs, ent_rep)  # (n,)
                best_idx = torch.argmax(scores).item()
                chosen_chunk_global_idx = cand_idx[best_idx]
                # expand it
                expansions[chosen_chunk_global_idx] = top[chosen_chunk_global_idx][1]['text']
                expansions_done += 1
                # heuristic break condition: if policy score is too low, stop expanding
                if float(scores[best_idx].cpu()) < 0.1:
                    break
            else:
                break
        else:
            # no policy or expansions exhausted => finish
            break

    return generated


# -----------------------
# Small training scaffold for policy (REINFORCE)
# -----------------------
def train_policy_simple(policy: PolicyNet, index: EmbeddingIndex, llm: LLMWrapper,
                        train_examples: List[Tuple[str, str]], epochs=10, lr=1e-4):
    """
    A toy training loop: for each (query, target_answer) we:
    - retrieve top chunks
    - sample an expansion action (probabilities via policy)
    - expand chosen chunks and compute reward as (LLM answer similarity to target)
    - update policy via REINFORCE
    NOTE: This is a minimal sketch for demonstration only.
    """
    opt = optim.Adam(policy.parameters(), lr=lr)
    import numpy as np
    for epoch in range(epochs):
        total_loss = 0.0
        for q, target in train_examples:
            top = index.query(q, k=TOP_K)
            # encode features
            q_emb = torch.tensor(index.encoder.encode([q]), dtype=torch.float32).to(DEVICE)
            candidates = [m['text'] for _, m in top]
            cand_embs = torch.tensor(index.encoder.encode(candidates), dtype=torch.float32).to(DEVICE)
            ent = torch.tensor([0.0], dtype=torch.float32).to(DEVICE)  # initial proxy
            # get logits and sample one chunk
            logits = policy(q_emb.repeat(cand_embs.size(0), 1), cand_embs, ent.repeat(cand_embs.size(0)))
            probs = torch.softmax(logits, dim=0)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            chosen = action.item()
            # build prompt with chosen expanded
            expansions_map = {chosen: candidates[chosen]}
            prompt = "Context:\n" + candidates[chosen] + "\n\nQuery: " + q + "\nAnswer:"
            ans = llm.generate(prompt, max_new_tokens=64)
            # compute reward: negative token-level distance or BLEU-like; here simple overlap
            reward = float(len(set(ans.split()) & set(target.split())) / max(1, len(target.split())))
            # policy loss: -logprob * reward
            logprob = torch.log(probs[action] + 1e-12)
            loss = -logprob * reward
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} loss {total_loss:.4f}")


# -----------------------
# Example usage
# -----------------------
def main_demo():
    # sample docs
    docs = {
        "doc1": "Historically, the capital of France is Paris. Paris is also known for the Eiffel Tower and the Louvre museum. It has a large population and is a center of art, fashion, and culture.",
        "doc2": "Python is a high-level programming language. Its syntax is easy to learn. Python is used for web development, data analysis, machine learning, and scripting.",
        "doc3": "Transformers are architectures for sequence modeling. They use self-attention and positional encodings. Large language models are built on transformer blocks."
    }
    # build index
    idx = EmbeddingIndex()
    idx.build_from_docs(docs)
    # init policy
    policy = PolicyNet(idx.encoder.get_sentence_embedding_dimension()).to(DEVICE)
    # init llm
    llm = LLMWrapper()
    # sample query
    q = "How is Paris culturally important?"
    answer = refrag_answer(q, idx, llm, policy=None)  # try with no policy first (heuristic)
    print("Answer (heuristic):\n", answer)

    # optional: train policy on toy examples then re-run
    train_examples = [(q, "Paris is a cultural center known for the Eiffel Tower, museums like the Louvre, art and fashion.")]
    train_policy_simple(policy, idx, llm, train_examples, epochs=2)
    answer2 = refrag_answer(q, idx, llm, policy=policy)
    print("Answer (with policy):\n", answer2)


if __name__ == "__main__":
    main_demo()
