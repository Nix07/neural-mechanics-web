# Neural Mechanics: Course Reading List

Curated readings for **Special Topics in AI: The Neural Mechanics of Concepts** at Northeastern University.

---

## Week 0: Introduction

*Course introduction, logistics, and context. No assigned readings.*

---

## Week 1: Foundations

| Paper | Why |
|-------|-----|
| Elhage et al. (2021) "[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)" | Residual stream view, vocabulary for mechanistic interpretability |
| Ferrando, Sarti, Bisazza & Costa-jussà (2024) "[A Primer on the Inner Workings of Transformer-based Language Models](https://arxiv.org/abs/2405.00208)" | Accessible pedagogical overview of interpretability techniques |
| nostalgebraist (2020) "[interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)" | Original logit lens technique — decode intermediate layers to vocabulary |
| Wendler et al. (2024) "[Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/abs/2402.10588)" | Logit lens reveals English as internal pivot language in multilingual models |
| Belrose et al. (2023) "[Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)" | Improved logit lens with learned affine probes per layer |

---

## Week 2: Steering

| Paper | Why |
|-------|-----|
| Li et al. (2023) "[Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/abs/2306.03341)" | Steer model behavior by adding vectors to activations |
| Zou et al. (2023) "[Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)" | Broader framework for reading and controlling representations |
| Elhage et al. (2022) "[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)" | Explains why features are entangled; essential background |
| Bricken et al. (2023) "[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)" *(optional)* | Background on SAE features explored via Neuronpedia |
| Templeton et al. (2024) "[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)" *(optional)* | SAE features at scale; context for Neuronpedia exploration |

---

## Week 3: Evaluation Methodology

| Paper | Why |
|-------|-----|
| Brown et al. (2020) "[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)" | Introduced few-shot evaluation paradigm |
| Petroni et al. (2019) "[Language Models as Knowledge Bases?](https://arxiv.org/abs/1909.01066)" | Template for probing factual knowledge via cloze tasks (LAMA) |
| Zheng et al. (2023) "[Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)" | Modern evaluation technique for open-ended outputs |
| Perez et al. (2022) "[Discovering Language Model Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251)" | Using LLMs to generate evaluation datasets |

---

## Week 4: Representation Geometry

| Paper | Why |
|-------|-----|
| Mikolov et al. (2013) "[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)" | Origin of "linear directions encode concepts" hypothesis (Word2Vec) |
| Bolukbasi et al. (2016) "[Man is to Computer Programmer as Woman is to Homemaker?](https://arxiv.org/abs/1607.06520)" | First demonstration of finding/manipulating concept directions |
| Marks & Tegmark (2023) "[The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets](https://arxiv.org/abs/2310.06824)" | Finds linear truth directions in LLMs |
| Tigges et al. (2023) "[Linear Representations of Sentiment in Large Language Models](https://arxiv.org/abs/2310.15154)" | Clean example of concept geometry in modern LLMs |
| Hernandez et al. (2023) "[Linearity of Relation Decoding in Transformer LMs](https://arxiv.org/abs/2308.09124)" | How relational concepts are encoded |

---

## Week 5: Causal Localization

| Paper | Why |
|-------|-----|
| Meng et al. (2022) "[Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)" | Causal tracing to localize where facts are stored (ROME) |
| Todd et al. (2023) "[Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213)" | How abstract functions are localized |
| Prakash et al. (2024) "[Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking](https://arxiv.org/abs/2402.14811)" | How models bind properties to entities |

---

## Week 6: Probes

| Paper | Why |
|-------|-----|
| Kim et al. (2018) "[Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors](https://arxiv.org/abs/1711.11279)" | Concept Activation Vectors (TCAV) — foundational probing method |
| Tenney et al. (2019) "[What Do You Learn from Context? Probing for Sentence Structure in Contextualized Word Representations](https://arxiv.org/abs/1905.06316)" | Systematic methodology for probing linguistic structure (Edge Probing) |
| Hewitt & Liang (2019) "[Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368)" | What probes actually tell you; addresses "is the probe just memorizing?" |

---

## Week 7: Attribution

| Paper | Why |
|-------|-----|
| Sundararajan et al. (2017) "[Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)" | Foundational input attribution method (Integrated Gradients) |
| Qi et al. (2024) "[Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation](https://arxiv.org/abs/2406.13663)" | Applied: attribution for understanding context usage in RAG (MIRAGE) |
| Gurrapu et al. (2023) "[Rationalization for Explainable NLP: A Survey](https://arxiv.org/abs/2301.08912)" *(optional)* | Map of the rationalization literature |

---

## Week 8: Circuits

| Paper | Why |
|-------|-----|
| Olsson et al. (2022) "[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)" | Canonical circuit analysis showing capability emergence |
| Conmy et al. (2023) "[Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997)" | Automated circuit discovery tool (ACDC) |
| Hanna, Pezzelle & Belinkov (2024) "[Have Faith in Faithfulness](https://arxiv.org/abs/2403.17806)" | EAP-IG: more faithful automated circuit discovery |

---

## Week 9: Decoding Representations & Self-Description

| Paper | Why |
|-------|-----|
| Ghandeharioun et al. (2024) "[Patchscopes: A Unifying Framework for Inspecting Hidden Representations](https://arxiv.org/abs/2401.06102)" | Use LLM's own abilities to decode its representations; unifies and extends logit lens |
| Hewitt, Geirhos & Kim (2025) "[We Can't Understand AI Using our Existing Vocabulary](https://arxiv.org/abs/2502.07586)" | Position paper: need neologisms for human-machine communication about concepts |
| Hewitt et al. (2025) "[Neologism Learning for Controllability and Self-Verbalization](https://arxiv.org/abs/2510.08506)" | Teach models new words for concepts; models self-verbalize what concepts mean to them |

---

## Summary

- **Total papers**: 34 (31 core + 3 optional)
- **Weeks covered**: 0–9 (Week 0 is intro, Weeks 1–9 have readings)
- **Focus**: Methods for localizing and characterizing concept representations in LLMs
