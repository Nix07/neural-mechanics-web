# Neural Mechanics: Course Reading List

Curated readings for **Special Topics in AI: The Neural Mechanics of Concepts** at Northeastern University.

---

## Week 0: Introduction

*Course introduction, logistics, and context. No assigned readings.*

---

## Week 1: Foundations

How can we peer inside a running language model to see what it's "thinking"? The key insight is that transformer intermediate layers encode evolving predictions that we can decode and inspect. This week introduces the conceptual vocabulary and core techniques for mechanistic interpretability.

**Ferrando, Sarti, Bisazza & Costa-jussà (2024)** "[A Primer on the Inner Workings of Transformer-based Language Models](https://arxiv.org/abs/2405.00208)" provides an accessible pedagogical overview of interpretability techniques. It surveys the landscape of methods we'll explore throughout the course, establishing shared vocabulary for discussing model internals.

**nostalgebraist (2020)** "[interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)" introduced the logit lens technique—the idea that we can decode intermediate layer activations directly into vocabulary space to see what the model is "predicting" at each layer. This simple but powerful idea opened a window into the progressive refinement of representations through the network.

**Wendler et al. (2024)** "[Do Llamas Work in English? On the Latent Language of Multilingual Transformers](https://arxiv.org/abs/2402.10588)" applies the logit lens to multilingual models, revealing a striking finding: regardless of input language, models often pivot through English in their intermediate representations before producing output in the target language. This demonstrates how interpretability tools can uncover unexpected computational strategies. *(For a refined version of logit lens with learned probes, see Belrose et al. 2023 "[Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112)".)*

*(Optional)* **Elhage et al. (2021)** "[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)" offers a deeper mathematical treatment of the residual stream view that underlies mechanistic interpretability. Dense but foundational for those who want the full theoretical grounding.

---

## Week 2: Steering

If concepts are encoded as directions in activation space, can we control model behavior by intervening on those directions? This week explores how to read and write to the model's internal representations—not just observe them, but manipulate them at inference time.

**Li et al. (2023)** "[Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/abs/2306.03341)" demonstrates that adding carefully chosen vectors to intermediate activations can steer models toward more truthful outputs. This establishes a practical paradigm: identify a direction that encodes a concept (like "truthfulness"), then add or subtract it during inference to control behavior.

**Zou et al. (2023)** "[Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405)" broadens this into a general framework for reading and controlling model representations. Rather than analyzing individual neurons, it treats high-dimensional activation patterns as the unit of analysis, enabling systematic manipulation of concepts like honesty, harmlessness, and reasoning style.

**Elhage et al. (2022)** "[Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html)" addresses a fundamental puzzle: why are features entangled in the first place? Through elegant toy experiments, this paper shows how neural networks learn to pack more features than they have dimensions by exploiting superposition—essential background for understanding why steering sometimes works and sometimes doesn't.

*(Optional)* **Bricken et al. (2023)** "[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html)" introduces sparse autoencoders (SAEs) as a tool for disentangling superposed features into interpretable components. This is background for exploring features via Neuronpedia.

*(Optional)* **Templeton et al. (2024)** "[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)" scales SAE analysis to Claude-scale models, demonstrating that interpretable features emerge even in frontier systems. Provides context for Neuronpedia exploration exercises.

---

## Week 3: Evaluation Methodology

How do we rigorously evaluate what models know and how they behave? Before we can study how concepts are represented, we need principled ways to test model capabilities and measure the effects of our interventions.

**Petroni et al. (2019)** "[Language Models as Knowledge Bases?](https://arxiv.org/abs/1909.01066)" established the template for probing factual knowledge via cloze-style tasks. The LAMA benchmark showed that language models encode substantial factual knowledge accessible through carefully designed prompts—a methodology we'll adapt for probing concept representations.

**Zheng et al. (2023)** "[Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685)" addresses the challenge of evaluating open-ended model outputs. As we study how models represent complex concepts, we need evaluation methods that go beyond exact-match accuracy. This paper establishes when and how to use LLMs themselves as evaluators.

**Perez et al. (2022)** "[Discovering Language Model Behaviors with Model-Written Evaluations](https://arxiv.org/abs/2212.09251)" demonstrates using LLMs to generate evaluation datasets at scale. This meta-technique is valuable for studying concept representations: we can use models to generate diverse examples of concept usage, then study how those examples are processed internally.

*(Optional)* **Brown et al. (2020)** "[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)" (GPT-3) introduced the few-shot evaluation paradigm that now dominates the field. Essential historical context for understanding modern evaluation practices.

---

## Week 4: Representation Geometry

What does a concept look like inside a neural network? This week examines the geometric structure of representations—visualizing activations with PCA, finding linear directions that correspond to concepts, and characterizing how abstract ideas like truth, sentiment, and relations are spatially organized in activation space.

**Marks & Tegmark (2023)** "[The Geometry of Truth: Emergent Linear Structure in LLM Representations of True/False Datasets](https://arxiv.org/abs/2310.06824)" demonstrates that truth and falsehood correspond to a linear direction in representation space. Across diverse factual statements, a simple linear probe can separate true from false—and this direction emerges naturally from model training without explicit supervision.

**Tigges et al. (2023)** "[Linear Representations of Sentiment in Large Language Models](https://arxiv.org/abs/2310.15154)" provides a clean case study of concept geometry, showing that sentiment (positive/negative valence) is encoded linearly in modern LLMs. The paper demonstrates practical techniques for finding and validating concept directions using difference-of-means and linear probes.

**Hernandez et al. (2023)** "[Linearity of Relation Decoding in Transformer LMs](https://arxiv.org/abs/2308.09124)" extends beyond unary properties to relational concepts. How does a model encode "the capital of X" or "the author of Y"? This paper shows that even these relational mappings are approximately linear, suggesting a unified geometric picture of concept encoding.

*(Optional)* **Mikolov et al. (2013)** "[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)" (Word2Vec) is the historical origin of the "linear directions encode concepts" hypothesis. The famous king − man + woman = queen example launched a research program that continues in today's LLM interpretability work.

*(Optional)* **Bolukbasi et al. (2016)** "[Man is to Computer Programmer as Woman is to Homemaker?](https://arxiv.org/abs/1607.06520)" demonstrated that concept directions could not only be found but manipulated—showing how to "debias" word embeddings by projecting out gender directions. An early template for the steering work we studied in Week 2.

---

## Week 5: Causal Localization

Where in the network are specific facts and functions computed? Correlation isn't causation: just because a representation encodes information doesn't mean that location is causally responsible for the model's behavior. This week introduces causal intervention methods that pinpoint where computations actually happen.

**Meng et al. (2022)** "[Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)" introduces causal tracing—a method for localizing where factual knowledge is stored by corrupting inputs and restoring activations at specific locations. The paper shows that factual associations are concentrated in middle-layer MLP modules, and demonstrates ROME, a technique for surgically editing facts.

**Todd et al. (2023)** "[Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213)" extends localization from facts to abstract functions. When a model performs in-context learning of a task (like "translate to French" or "answer with antonyms"), where is that function encoded? This paper shows that task-specific "function vectors" are localized and transferable.

**Prakash et al. (2024)** "[Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking](https://arxiv.org/abs/2402.14811)" examines how models bind properties to entities—a fundamental challenge for representing concepts about objects. The paper traces how entity-tracking mechanisms develop and localize through training.

---

## Week 6: Probes

Is the information actually there? Probing asks the information-content question: can we reliably decode a concept from activations, how cleanly is it encoded, at which layers, and how do we know the probe is revealing genuine structure rather than imposing its own? This week covers the methodology and pitfalls of training classifiers to read out concepts.

**Kim et al. (2018)** "[Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors](https://arxiv.org/abs/1711.11279)" introduces TCAV (Testing with Concept Activation Vectors), a foundational probing method. Rather than asking "which input features matter," TCAV asks "how sensitive is the model to this human-defined concept?" This reframes interpretability around concepts meaningful to humans.

**Hewitt & Liang (2019)** "[Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368)" addresses a critical question: when a probe succeeds, what does that tell us? A sufficiently powerful probe might memorize patterns rather than reveal genuine representations. This paper introduces control tasks and selectivity measures to ensure probes are informative.

*(Optional)* **Tenney et al. (2019)** "[What Do You Learn from Context? Probing for Sentence Structure in Contextualized Word Representations](https://arxiv.org/abs/1905.06316)" presents the "edge probing" methodology for systematically studying what linguistic structures are encoded across layers. A template for thorough, rigorous probing studies.

---

## Week 7: Attribution

Which parts of the input drive specific model behaviors? Attribution methods trace the causal path from inputs through computations to outputs, helping us understand how models use their context—and whether they're using it appropriately.

**Sundararajan et al. (2017)** "[Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365)" establishes Integrated Gradients, a principled input attribution method grounded in axiomatic requirements (sensitivity, implementation invariance). Unlike simple gradient methods, it handles saturation and provides theoretically grounded attributions.

**Qi et al. (2024)** "[Model Internals-based Answer Attribution for Trustworthy Retrieval-Augmented Generation](https://arxiv.org/abs/2406.13663)" applies attribution to a pressing practical problem: when a RAG system generates an answer, which retrieved documents actually contributed? MIRAGE uses internal model signals to provide faithful attribution, revealing when models ignore or misuse their context.

*(Optional)* **Gurrapu et al. (2023)** "[Rationalization for Explainable NLP: A Survey](https://arxiv.org/abs/2301.08912)" surveys the rationalization literature—methods that generate natural language explanations of model decisions. Provides a map of the broader landscape connecting attribution to explanation.

---

## Week 8: Circuits

Can we reverse-engineer the end-to-end algorithms implemented by a neural network? Circuit analysis aims to understand the mechanism for a prediction from beginning to end—not just isolated features, but the specific attention heads and MLPs that work together to implement a function.

**Olsson et al. (2022)** "[In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)" is the canonical circuit analysis, showing how "induction heads" (attention patterns that match and copy previous sequences) emerge during training and drive in-context learning capabilities. This paper establishes the methodology of tracing capabilities to specific, interpretable circuits.

**Conmy et al. (2023)** "[Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997)" introduces ACDC (Automatic Circuit DisCovery), addressing the scalability challenge of manual circuit analysis. By automatically identifying which components are necessary for a behavior, ACDC enables circuit analysis on larger models and more complex tasks.

**Hanna, Pezzelle & Belinkov (2024)** "[Have Faith in Faithfulness](https://arxiv.org/abs/2403.17806)" improves on automated circuit discovery with EAP-IG (Edge Attribution Patching with Integrated Gradients). The method produces more faithful circuits by combining gradient-based attribution with causal intervention, addressing limitations of earlier automated approaches.

---

## Week 9: Decoding Representations & Self-Description

Can models help us interpret themselves? This week explores techniques that leverage the model's own capabilities to decode and describe its internal representations—turning the model into a collaborator in the interpretability process.

**Ghandeharioun et al. (2024)** "[Patchscopes: A Unifying Framework for Inspecting Hidden Representations](https://arxiv.org/abs/2401.06102)" uses the model itself to decode its representations. By "patching" hidden states into carefully designed prompts, Patchscopes elicits natural language descriptions of what information is encoded at any layer and position. This unifies and extends the logit lens approach from Week 1.

**Hewitt et al. (2025)** "[Neologism Learning for Controllability and Self-Verbalization](https://arxiv.org/abs/2510.08506)" teaches models new words for concepts they already represent internally. When a model learns a neologism for an internal concept, it can then describe what that concept means to it—enabling a form of self-report about representations. This opens new possibilities for human-machine communication about internal states.

*(Optional)* **Hewitt, Geirhos & Kim (2025)** "[We Can't Understand AI Using our Existing Vocabulary](https://arxiv.org/abs/2502.07586)" is a position paper arguing that effective human-AI communication requires developing new shared vocabulary for concepts that exist in models but not in natural language. Provides motivation for the neologism learning approach.

---

## Summary

- **Total papers**: 34 (22 core + 12 optional)
- **Weeks covered**: 0–9 (Week 0 is intro, Weeks 1–9 have readings)
- **Focus**: Methods for localizing and characterizing concept representations in LLMs
