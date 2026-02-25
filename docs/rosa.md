# RWKV-8 ROSA: Beyond Attention

**Date:** October 11, 2025 | Updated: January 20, 2026

---

## Introduction
This document details the **RWKV-8 ROSA** mechanism, a novel neurosymbolic approach designed to replace traditional attention mechanisms in neural networks. **ROSA (Rapid Online Suffix Automaton)** enables infinite-range, lossless information propagation by efficiently processing discrete token sequences.

---

## Core Principle
Given a token sequence \( x = x_0x_1 \dots x_{n-1} \), ROSA computes a new sequence \( y = y_0y_1 \dots y_{n-1} \) using suffix automata logic. For each position \( i \), the output \( y_i \) is defined as:

\[
y_i=\begin{cases}
x_{j+1}, & \text{if there exists } j < i \text{ and } m \geq 0 \text{ s.t. } x_{j-m:j}=x_{i-m:i}\\
-1, & \text{otherwise}
\end{cases}
\]

Here, \( j \) is chosen to maximize the matching length \( m \), with ties broken by selecting the largest \( j \).

The algorithm iterates through tokens, maintaining state transitions to track contextual dependencies. It features adaptive time complexity, typically \( O(n) \) but \( O(n^2) \) in the worst case.

---

## Performance & Applications
- **Prediction Accuracy**: ROSA's error rate decreases as context length increases, outperforming naive baselines on long-document tasks (tested on 10,000+ token sequences from "The Pile").
- **Practical Use Cases**:
  - Integrate \( \text{Emb}(\text{ROSA}(x)) \) into LLM layers (especially early ones) for enhanced contextual understanding. This method is also valid for kNN+LLM and useful for RAG.
  - Employ neurosymbolic integration: Add \( \text{Emb}(\text{ROSA}(\text{Sampling}(z_a))) \) to tensor \( z_b \) (e.g., in the next layer). This allows the LLM+ROSA to invent its own inner monologue languages. Each LLM layer can produce multiple sequences with small vocabularies for fast parallel ROSA processing.
  - Enable dynamic token bookmarking (e.g., `<C_begin><C_3><C_6><C_end>`) to mark the start of a specific chat round, allowing ROSA to perfectly retrieve any chat history. This supports speculative decoding for speedup and can be managed end-to-end by the LLM.

---

## Advantages Over Attention
ROSA operates directly on discrete tokens, avoiding computationally expensive operations like dot products and softmax. This design eliminates the need for a KV cache, significantly reducing memory overhead and enabling real-time, scalable inference on both CPU and GPU architectures.

---

## ROSA Variants
Several architectural variants of the ROSA mechanism have been explored, offering different trade-offs in simplicity, memory, and performance.

### 1. **ROSA-QKV-1bit**
This is currently the best overall design. It quantizes the standard Q/K/V float tensors to 1 bit (e.g., values > 0 map to 1, ≤ 0 map to 0). The ROSA algorithm matches each 1-bit query sequence \( Q_i \) within its corresponding key sequence \( K_i \). The output value is taken from the 1-bit \( V_i \) at the matched position (or set to 0 if unmatched). Finally, a trainable per-channel vector \( e_i \) scales the binary output (1 → \( e_i \), 0 → \( -e_i \)). This 1-bit quantization enables very efficient gradient computation and learning.

### 2. **ROSA-QKV-floatV**
An intuitive extension is to use a full-precision (float) value tensor \( V \) and remove the scaling vector \( e_i \). However, initial experiments surprisingly show worse performance. This may be because the specific gradient computation methods developed for the 1-bit variant, combined with learning \( e_i \) and a binary \( V \), are more effective than learning a continuous \( V_\text{float} \).

### 3. **ROSA-QKV-lite**
An extremely simple discrete memory system. It uses a larger vocabulary and simply matches the last occurrence of a query token \( Q \) in the key sequence \( K \). This is equivalent to limiting the ROSA matching length to 1, prioritizing speed and simplicity over contextual depth.

### 4. **ROSA-QKV-NNS**
This variant departs from exact ROSA matching. It uses full-precision Q/K vectors and performs a nearest-neighbor search (e.g., using a k-d tree for speed) to find the closest key for each query, acting like a top-1 attention mechanism. This could be promising for improving long-context performance in RNNs, as it maintains a small, efficient KV cache that can be computed on a CPU.

### 5. **ROSA-RePo**
Inspired by methods like [RePo](https://pub.sakana.ai/repo/), this explores reordering ROSA tokens for efficiency. The primary challenge lies in computing gradients efficiently through this reordering operation.

### 6. **ROSA-4bit Language Model Implementation**
The provided code implements a production-ready **4-bit ROSA language model** within the RWKV-8 architecture ("Heron"). This design strikes a balance between the efficiency of 1-bit quantization and representational capacity.

**Core Mechanism (`rosa_slow_4bit_layer`):**
*   **Grouping:** The model channels (C) are grouped into blocks of 4 (or more generally, `bits`). For each group `g`, a 4-bit symbol is constructed per timestep by packing the 1-bit values from its constituent channels.
*   **Matching:** The ROSA algorithm operates on these sequences of 4-bit symbols. It matches the query symbol sequence \( Q_\text{sym} \) against the key symbol sequence \( K_\text{sym} \) to find the longest suffix match.
*   **Output:** For a matched position \( i \), the corresponding 4-bit value symbol \( V_\text{sym}[i] \) is retrieved. This symbol is unpacked, and each bit controls the output for its corresponding channel: `bit=1` outputs `+e[channel]`, `bit=0` outputs `-e[channel]`. Unmatched positions output `0.0`.
*   **Integration:** The `RWKV_ROSA_4bit` module integrates this layer into a standard Q/K/V projection setup with time-shift mixing, similar to other RWKV components.

**Model Architecture:**
The model follows a standard pre-normalization Transformer-style block:
1.  **LayerNorm:** Applied to the input.
2.  **ROSA-4bit Attention:** The core `RWKV_ROSA_4bit` module replaces the standard attention mechanism.
3.  **Channel-wise FFN (`RWKV_CMix_x070`):** A feed-forward network with ReLU² activation, applied after another LayerNorm.
4.  Residual connections are added around both the ROSA and FFN sub-blocks.

**Architecture Flowchart**

```mermaid
flowchart TD
    A[Input Tokens] --> B[Token Embedding]
    B --> C[Block Processing<br/>n_layers]
    
    subgraph C [Block Processing]
        D[LayerNorm] --> E[ROSA-4bit Module]
        E --> F[Residual Connection]
        F --> G[LayerNorm]
        G --> H[Channel-wise FFN<br/>RWKV_CMix_x070]
        H --> I[Residual Connection]
    end
    
    I --> J[Repeat for n_layers]
    J --> K[Final LayerNorm]
    K --> L[Output Head]
    L --> M[Next Token Logits]
    
    subgraph E [ROSA-4bit Module]
        N[Time Shift Mixing<br/>q = x + xx * x_q<br/>k = x + xx * x_k<br/>v = x + xx * x_v]
        N --> O[Linear Projection<br/>Q = Linear q<br/>K = Linear k<br/>V = Linear v]
        O --> P[1-bit Quantization<br/>Q_b = Q &gt; 0<br/>K_b = K &gt; 0<br/>V_b = V &gt; 0]
        P --> Q[4-bit Group Packing]
        
        subgraph Q [4-bit Group Packing]
            R[Group channels into<br/>groups of 4<br/>g = C // 4]
            R --> S["Pack bits per group:<br/>sym = bit0 OR bit1 SHL 1 OR bit2 SHL 2 OR bit3 SHL 3"]
        end
        
        Q --> T[ROSA Matching per Group]
        
        subgraph T [ROSA Matching per Group]
            U[For each position i in sequence]
            U --> V[Find longest suffix match<br/>of Q_sym in K_sym<br/>before position i]
            V --> W["If match found:<br/>output = V_sym at matched_index<br/>Else: output = -1"]
        end
        
        T --> X[Bit Unpacking & Scaling]
        
        subgraph X [Bit Unpacking & Scaling]
            Y["For each bit in 4-bit output:<br/>bit = output SHR b AND 1"]
            Y --> Z["If matched:<br/>bit=1 → +e[channel]<br/>bit=0 → -e[channel]<br/>Else: 0.0"]
        end
        
        X --> AA[Linear Projection]
    end
    
    subgraph H [Channel-wise FFN]
        AB[Time Shift Mixing] --> AC[Linear Layer] --> AD[ReLU² Activation] --> AE[Linear Layer]
    end
```

---

## Rust Implementation Details

This section describes the concrete implementation of ROSA using Suffix Automaton (SAM) in Rust.

### SAM State Structure

Each state in the suffix automaton maintains:

```rust
struct State<const S: usize> {
    /// Transition map: token → state index.
    next: Box<[Option<NonMaxUsize>; S]>,
    /// Suffix link (points to a state representing a proper suffix).
    link: Option<NonMaxUsize>,
    /// Length of the longest string in this equivalence class.
    len: usize,
    /// The last position of the state occurrence in the sequence.
    end: Option<NonMaxUsize>,
}
```

### SAM Structure

```rust
pub struct Sam<T, const S: usize> {
    /// All states in the automaton.
    states: Vec<State<S>>,
    /// The last state added.
    last: usize,
    /// The current sequence of tokens.
    sequence: Vec<T>,
}
```

### Core Algorithm: Token Insertion

When inserting a new token `c`, the algorithm follows these steps:

1. **Create a new state** `cur` with `len = states[last].len + 1`
2. **Add transitions** from all suffix states of `last` that don't have a transition on `c`
3. **Handle conflicts**: If a state `p` already transitions on `c` to state `q`:
   - **Simple case**: If `states[p].len + 1 == states[q].len`, set `link[cur] = q`
   - **Clone case**: Otherwise, clone state `q` to `clone` with adjusted `len`, update suffix links accordingly

```rust
fn push_internal(&mut self, token: T) {
    let end = self.sequence.len();
    let current = self.states.len();
    self.states.push(State {
        link: None,
        len: self.states[self.last].len + 1,
        ..Default::default()
    });

    // add transitions from suffix chain
    // handle conflicts with clone operation
    // update suffix links
    // ...
}
```

### Incremental Matching

The key innovation for online processing is `match_end_incremental`, which maintains state across calls:

```rust
pub fn match_end_incremental(&self, token: T, state: usize) -> (Option<usize>, usize) {
    let token = token.into();
    let mut p = state;
    
    // traverse suffix links until transition found
    while self.states[p].next[token].is_none()
        && let Some(link) = self.states[p].link.map(Into::into)
    {
        p = link;
    }

    match self.states[p].next[token].map(usize::from) {
        Some(next) => (self.states[next].end.map(Into::into), next),
        None => (None, 0),
    }
}
```

### ROSA Structure

```rust
pub struct Rosa<T, V, const S: usize> {
    /// Key tokens form a SAM.
    ks: Sam<T, S>,
    /// Value tokens.
    vs: Vec<V>,
    /// Current state of longest matched query suffix in keys.
    state: usize,
}
```

### ROSA Push Operation

The core `push` method implements the online ROSA algorithm:

```rust
pub fn push(&mut self, q: T, k: T, v: V) -> Option<V> {
    // match query token against existing key SAM
    let (end, state) = self.ks.match_end_incremental(q, self.state);
    self.state = state;

    // add new key-value pair
    self.ks.push(k);
    self.vs.push(v);

    // return value at matched position (if any)
    end.map(|index| self.vs[index])
}
```

### Time Complexity Analysis

| Operation | Average | Worst Case |
|-----------|---------|------------|
| SAM construction (per token) | O(1) amortized | O(n) |
| Pattern matching | O(m) | O(m) |
| Incremental matching (per token) | O(1) amortized | O(n) |
| Full ROSA sequence (n tokens) | O(n) | O(n²) |

The worst case O(n²) occurs when the sequence has highly repetitive patterns with many suffix link traversals. In practice, for natural language and typical model outputs, performance is close to O(n).

### Memory Efficiency

The SAM typically uses at most `2n - 1` states for a sequence of length `n`. The `NonMaxUsize` type is used for memory optimization, saving 1 bit per state index by utilizing the fact that `usize::MAX` is reserved as a sentinel value.

---

## Conclusion
RWKV-8 ROSA bridges symbolic and neural paradigms, offering a parameter-free, neurosymbolic solution for next-token prediction and beyond. Its efficiency, adaptability, and lossless information propagation position it as a robust alternative to attention mechanisms in modern AI workflows. The ongoing exploration of variants (1-bit, 4-bit, NNS, etc.) and its successful integration into a full language model underscore its versatility and potential for efficient, long-context reasoning.
