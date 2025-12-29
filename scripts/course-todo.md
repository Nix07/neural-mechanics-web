# Neural Mechanics: Course Development TODO

Open items to revisit as the course materials are developed.

---

## Pending External Dependencies

- [ ] **Neuronpedia steering on larger models** — Emailed to ask if steering can be enabled on GPT-OSS-20B or other larger models. If enabled, the Week 2 "find a pun feature" exercise becomes much stronger.
  - Current steering-enabled models: Gemma-2B, Gemma-2B-IT, GPT2-Small
  - Desired: GPT-OSS-20B (has nice pun feature at `15-resid-post-aa/37788`)

---

## Exercises to Write

- [ ] **Week 1 Exercise: Logit lens and the Wendler result**
  - Reproduce the finding that multilingual models pivot through English
  - Use logit lens to decode intermediate layers
  - Observe three-phase pattern: ambiguous → English-dominant → target language

- [ ] **Week 2 Exercise: Neuronpedia concept steering**
  - Systematic workflow: search by activation, search by explanation, validate, steer
  - "Can you find a pun feature?" as motivating example
  - Design to work with current models, update if larger models become available
  - Include: what to do if you can't find a feature (that's also a valid finding)

- [ ] **Running In-Class Exercise: Puns/humor thread**
  - A single concept explored with each week's methods — demonstrates how techniques build on each other
  - Week 2: Find pun/humor features on Neuronpedia, try steering
  - Week 3: Create a pun evaluation dataset; score models on pun understanding/completion
  - Week 4: Visualize pun vs non-pun representations; find a "pun direction" via difference-in-means
  - Week 5: Where is humor localized? (causal tracing)
  - Week 6: Train a probe for "is this a pun?"
  - Week 7: Attribution — which input words matter for pun recognition? Does the model focus on the double-meaning word?
  - Week 8: Attempt automated circuit discovery for puns using EAP-IG (Hanna et al.)
  - Week 9: Use Patchscopes to decode pun representations; try neologism learning for "pun" concept
  - By end of course: students have applied every major method to one concept
  - Format: ~20-30 min in-class activity each week, results carry forward

---

## Readings to Verify

- [ ] Double-check all arxiv links still work
- [ ] Verify transformer-circuits.pub links are correct
- [ ] Consider adding publication venues (NeurIPS, ICML, etc.) to readings list

---

## Future Additions to Consider

- [ ] Week-by-week coding exercises/notebooks
- [ ] Project milestone assignments tied to weekly topics
- [ ] Guest lecture schedule (if applicable)
- [ ] Links to relevant codebases (nnsight, TransformerLens, SAELens, Inseq, etc.)

---

## Notes

*Last updated: 2025-12-29*
