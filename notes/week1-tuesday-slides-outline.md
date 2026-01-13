# Week 1 Tuesday Lecture: Foundations & Logit Lens
## Slide Outline

---

### Opening (5 min)

**Slide 1: Title**
- "Foundations: Looking Inside Language Models"
- Week 1, Tuesday

**Slide 2: The Research Process is Iterative**
- Diagram: Idea → Feasibility Check → Refine → Experiment → Interpret → New Questions
- "Before you can ask an *interesting* question, you need to know what's *feasible*"
- Today: Building your toolkit for feasibility

**Slide 3: Today's Goals**
1. Whirlwind tour: Neural nets → Language models → Transformers
2. First interpretability method: The Logit Lens
3. Hands-on setup: HuggingFace + NDIF tokens
4. Lab: Reproduce experiments, discuss what they tell us

---

### Part 1: Whirlwind Review (20-25 min)

**Slide 4: Neural Networks in 60 Seconds**
- Input → Hidden layers → Output
- Each layer: linear transformation + nonlinearity
- Learning = adjusting weights to minimize loss
- Key insight: intermediate layers learn *representations*

**Slide 5: Language Modeling**
- Task: Predict the next token
- P(next token | previous tokens)
- Training: maximize likelihood on huge text corpora
- "The cat sat on the ___" → mat (0.3), floor (0.2), couch (0.15)...

**Slide 6: Tokens, Not Words**
- Tokenization: "understanding" → ["under", "standing"] or ["understand", "ing"]
- Vocabulary: ~32K-128K tokens
- Includes words, subwords, punctuation, special tokens
- Show example tokenization

**Slide 7: The Transformer Architecture**
- Diagram: Input embeddings → N layers → Output logits
- Each layer: Attention + MLP (feed-forward)
- Residual connections: information flows AND accumulates

**Slide 8: The Residual Stream View**
- Think of it as a "river" of information
- Each layer reads from and writes to the stream
- Hidden state at layer L = sum of all contributions so far
- This view is key for interpretability!

**Slide 9: Three Phases of a Forward Pass**
- Encoder: tokens → embeddings (input space)
- Transformer layers: embeddings → rich representations (concept space)
- Decoder: representations → vocabulary logits (output space)
- [Use the transformer-state-grid diagram]

**Slide 10: Key Components to Know**
| Component | What it does |
|-----------|--------------|
| Embedding | token ID → vector |
| Attention | tokens communicate |
| MLP | per-token processing |
| LayerNorm | normalize activations |
| Unembedding | vector → vocabulary logits |

---

### Part 2: The Logit Lens (20 min)

**Slide 11: The Key Question**
- We see inputs and outputs
- But what happens *in between*?
- Can we peek at intermediate "thoughts"?

**Slide 12: The Logit Lens Idea**
- At each layer, we have a hidden state
- What if we decoded it early?
- Apply the final layer norm + unembedding to intermediate states
- See what the model "would predict" at each layer

**Slide 13: Logit Lens Diagram**
- [Use the transformer-early-exit-logit-lens diagram]
- Show hidden states at each layer being projected to vocabulary
- Example: "The capital of France is ___"
  - Layer 0: "the" (0.1)
  - Layer 20: "Paris" (0.3)
  - Layer 40: "Paris" (0.6)
  - Layer 79: "Paris" (0.85)

**Slide 14: What Can We See?**
- When does the answer "crystallize"?
- Does meaning emerge gradually or suddenly?
- Are there "concept" layers vs "formatting" layers?

**Slide 15: Example - Multilingual Concepts**
- Prompt: "Espanol: amor, Francais: ___"
- Wendler et al. finding: English appears in middle layers!
- Early layers: input language encoding
- Middle layers: "concept space" (English-biased)
- Late layers: output language formatting

**Slide 16: Example - Puns**
- "Why do electricians make good swimmers? Because they know the ___"
- Track P("current") across layers
- When does the model "get" the joke?

**Slide 17: Limitations of Logit Lens**
- Observation ≠ Explanation
- We see *what* but not *why*
- Correlation vs causation
- The unembedding matrix wasn't trained for intermediate layers
- → Tuned Lens tries to address this

---

### Part 3: Getting Set Up (15-20 min)

**Slide 18: What You Need**
1. **HuggingFace account** → HF_TOKEN
   - huggingface.co → Settings → Access Tokens
2. **NDIF account** → NDIF_API_KEY
   - ndif.us → Sign up → Get API key
3. **Google Colab** (or local Jupyter)

**Slide 19: Setting Up Colab Secrets**
- Colab → Settings (gear icon) → Secrets
- Add: `HF_TOKEN` = your token
- Add: `NDIF_API_KEY` = your key
- These are auto-detected by nnsight

**Slide 20: Open the Lab Notebook**
- Link to: `labs/week1/logit_lens.ipynb`
- [![Open in Colab badge]]
- Run the setup cells
- Verify: "Model loaded" + 80 layers

**Slide 21: Troubleshooting**
- "API key not found" → Check secret names exactly
- "Model not available" → Check ndif.us status page
- Timeout → NDIF queue may be busy, retry

---

### Part 4: Lab & Discussion (30-40 min)

**Slide 22: Lab Exercises**
1. Run the "Capital of France" example
2. Try the multilingual example - do you see English emerge?
3. Try the pun examples - when does meaning shift?
4. Try your own prompts!

**Slide 23: Discussion Questions**
As you explore, think about:
- What counts as "evidence" from logit lens?
- If we see X in the middle layers, what can we conclude?
- What *can't* we conclude?

**Slide 24: The Evidence Question**
- Wendler et al. claim: "Models think in English"
- Evidence: English tokens peak in middle layers for non-English I/O
- But: Is this *because* the model "thinks" in English?
- Or: Is this an artifact of the unembedding matrix?
- Key: We need *interventions* to establish causation

**Slide 25: Representation Hijacking Demo**
- New finding: Yona et al. (2024) "Doublespeak"
- Context can shift word meanings across layers
- "Carrot" → "bomb" semantics with right context
- Logit lens reveals the layer-by-layer shift
- Implications for safety

**Slide 26: From Observation to Experiment**
- Logit lens = observation tool
- Raises questions, doesn't answer them
- Next steps: activation patching, causal interventions
- Your project: What observations lead to testable hypotheses?

---

### Closing (5 min)

**Slide 27: For Thursday**
- **Project Pitches Due**
- 1-2 page pitch document
- 5-minute presentation
- FINER framework: Feasible, Interesting, Novel, Ethical, Relevant
- Think: What concept? What method? What would we learn?

**Slide 28: Key Takeaways**
1. Transformers = residual stream + attention + MLPs
2. Logit lens = early exit decoding at each layer
3. Observation ≠ causation (but observations generate hypotheses)
4. You now have tools to peek inside 70B parameter models!

**Slide 29: Resources**
- Lab notebook: [link]
- Readings: Primer, Logit Lens blog, Latent Language paper
- NDIF status: ndif.us
- Course Discord / Office hours

---

## Timing Estimate
- Opening: 5 min
- Part 1 (Review): 20-25 min
- Part 2 (Logit Lens): 20 min
- Part 3 (Setup): 15-20 min
- Part 4 (Lab): 30-40 min
- Closing: 5 min
- **Total: ~100-115 min** (adjust lab time as needed)

## Diagrams Needed
1. Research iteration cycle
2. Neural network basics
3. Transformer architecture (residual stream view)
4. Three-phase model (encoder/layers/decoder)
5. Logit lens early exit visualization
6. Multilingual concept trajectories

## Notes
- Have the Colab notebook ready to demo live
- Prepare backup screenshots in case NDIF is slow
- Consider having TAs help with setup issues during Part 3
