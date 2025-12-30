# Neural Mechanics: Course Development TODO

Open items to revisit as the course materials are developed.

---

## Pending External Dependencies

- [ ] **Neuronpedia steering on larger models** — Emailed to ask if steering can be enabled on GPT-OSS-20B or other larger models. If enabled, the Week 2 "find a pun feature" exercise becomes much stronger.
  - Current steering-enabled models: Gemma-2B, Gemma-2B-IT, GPT2-Small
  - Desired: GPT-OSS-20B (has nice pun feature at `15-resid-post-aa/37788`)

---

## In-Class Exercise Preparation

### Week 2: Neuronpedia Pun Feature Search and Steering
- [x] **Exercise added to week2.html**
- [ ] Create Colab notebook with:
  - Code to load Neuronpedia feature vectors
  - Steering implementation using nnsight or TransformerLens
  - Example workflow: search → validate → steer
- [ ] Pre-test: verify humor/pun features are findable on Neuronpedia
- [ ] Backup plan: if no good features found, document "negative result" workflow

### Week 3: Pun Evaluation Dataset Creation
- [x] **Exercise added to week3.html**
- [ ] Create Colab notebook with:
  - Prompt templates for generating pun datasets
  - API setup for GPT-4 as evaluator
  - Logit lens visualization code
- [ ] Prepare seed examples: 10-15 diverse puns for students to start from
- [ ] Test: verify logit lens shows interesting patterns on puns

### Week 4: Pun Representation Visualization
- [x] **Exercise added to week4.html**
- [ ] Create Colab notebook with:
  - Activation extraction across layers
  - PCA visualization with color-coding
  - Mass-mean-difference direction computation
  - Classification accuracy using the direction
- [ ] Pre-extract: prepare dataset of ~50 puns and ~50 non-puns
- [ ] Test: verify PCA shows separation at some layer

### Week 5: Pun Causal Localization
- [x] **Exercise added to week5.html**
- [ ] Create Colab notebook with:
  - Causal tracing infrastructure (noise + restore)
  - Heatmap visualization (layer x position)
  - Pun-specific patching experiments
- [ ] Prepare: minimal pairs (pun + matched non-pun)
- [ ] Test: run causal tracing on 2-3 puns to verify signal

### Week 6: Pun Probe Training
- [x] **Exercise added to week6.html**
- [ ] Create Colab notebook with:
  - Probe training pipeline (logistic regression)
  - Layer-wise accuracy computation
  - Control task implementations (random labels, selectivity)
  - Comparison with Week 4 mean-difference direction
- [ ] Test: verify probes achieve reasonable accuracy

### Week 7: Pun Attribution Analysis
- [x] **Exercise added to week7.html**
- [ ] Create Colab notebook with:
  - Integrated gradients implementation (or use Inseq)
  - Attribution heatmap visualization
  - Method comparison (IG vs Input x Gradient vs Attention)
  - Ablation validation of attributions
- [ ] Test: verify attributions highlight punchline words

### Week 8: Pun Circuit Discovery
- [x] **Exercise added to week8.html**
- [ ] Create Colab notebook with:
  - EAP-IG implementation for head importance
  - Path patching validation code
  - Circuit visualization tools
  - Synthesis template (gather findings from weeks 2-7)
- [ ] Test: run EAP-IG on pun recognition task

### Week 9: Pun Emergence Over Training (NDIF/OLMo)
- [x] **Exercise added to week9.html**
- [ ] Create Colab notebook with:
  - NDIF connection and OLMo checkpoint access
  - Pun completion accuracy measurement across checkpoints
  - Probe accuracy tracking over training
  - Logit lens comparison at different training stages
- [ ] Test: verify NDIF access works and checkpoints are available
- [ ] Prepare: list of recommended OLMo checkpoint steps

### Week 10: Patchscopes for Puns
- [x] **Exercise added to week10.html**
- [ ] Create Colab notebook with:
  - Patchscopes implementation (hidden state patching)
  - Multiple decoder prompts
  - Layer comparison workflow
- [ ] Test: verify Patchscopes produces interpretable outputs for puns

---

## Exercises to Write

- [x] **Week 1 Exercise: Logit lens, Wendler result, and puns** (already in week1.html)
  - Reproduce the finding that multilingual models pivot through English
  - Pun analysis with logit lens (intro to pun thread)
  - Uses NDIF Logit Lens Workbench

- [x] **Week 2 Exercise: Neuronpedia concept steering** (structure added, notebook pending)

- [x] **Running In-Class Exercise: Puns/humor thread** (all weeks structured)
  - Week 1: Logit lens on puns — intro to the research question
  - Week 2: Find pun/humor features on Neuronpedia, try steering
  - Week 3: Create a pun evaluation dataset; score models on pun understanding/completion
  - Week 4: Visualize pun vs non-pun representations; find a "pun direction" via difference-in-means
  - Week 5: Where is humor localized? (causal tracing)
  - Week 6: Train a probe for "is this a pun?"
  - Week 7: Attribution — which input words matter for pun recognition?
  - Week 8: Attempt automated circuit discovery for puns using EAP-IG
  - Week 9: Track pun emergence over OLMo training checkpoints (NDIF)
  - Week 10: Use Patchscopes to decode pun representations
  - By end of course: students have applied every major method to one concept
  - Format: ~60 min in-class activity each week (3 parts of ~15-25 min), results carry forward

---

## Colab Notebook Development Checklist

All notebooks should include:
- [ ] Clear setup instructions (pip installs, model downloads)
- [ ] GPU detection and fallback to smaller models
- [ ] Pre-computed results for students without GPU access
- [ ] Links back to the week's HTML page
- [ ] Connection to previous week's results (cumulative pun thread)

Priority order for notebook development:
1. Week 3 (foundational dataset creation)
2. Week 4 (visualization, builds on Week 3)
3. Week 2 (can be standalone, Neuronpedia exploration)
4. Week 5 (builds on Week 4)
5. Week 6 (builds on Weeks 4-5)
6. Week 7 (builds on Week 6)
7. Week 8 (synthesis of all previous)
8. Week 9 (training dynamics with NDIF/OLMo)
9. Week 10 (capstone pun exercise)

---

## Readings to Verify

- [ ] Double-check all arxiv links still work
- [ ] Verify transformer-circuits.pub links are correct
- [ ] Consider adding publication venues (NeurIPS, ICML, etc.) to readings list

---

## Future Additions to Consider

- [ ] Week-by-week coding exercises/notebooks (in progress)
- [x] Project milestone assignments tied to weekly topics
- [x] Guest lecture schedule (Week 13 Tuesday)
- [ ] Links to relevant codebases (nnsight, TransformerLens, SAELens, Inseq, etc.)

---

## Notes

*Last updated: 2025-12-30*
