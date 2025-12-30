# Week 0: Are Concepts Different from Words?

Lecture notes for **Special Topics in AI: The Neural Mechanics of Concepts** at Northeastern University.

---

## The Central Question

Large language models process text, but do they think in words? This course begins with a provocation: the internal representations that drive model behavior may be fundamentally different from the tokens that flow in and out. If true, this has profound implications for how we understand, control, and extract knowledge from these systems.

Words are surface phenomena. They vary across languages, shift meaning with context, and often fail to capture the abstractions that underlie coherent reasoning. Concepts, by contrast, are the invariants&mdash;the stable structures that persist across linguistic transformations. This course asks: can we find these invariants inside neural networks? Can we characterize them, manipulate them, and ultimately build an atlas of the concepts that drive AI behavior?

---

## The Mysteries of Language Models

Modern LLMs exhibit capabilities that seem to exceed mere pattern matching over tokens. Consider in-context learning (ICL): a model that has never been trained on a specific task can perform it after seeing just a few examples in its prompt. This is a form of metareasoning&mdash;the model is not just processing text but reasoning about how to reason.

Where does this capability come from? It is not explicitly programmed. It emerges from training on next-token prediction, yet manifests as something that looks remarkably like flexible cognition. Understanding how such emergent capabilities are implemented&mdash;not just that they exist, but how they work mechanistically&mdash;is one of the core challenges that motivates interpretability research.

---

## The Gap Between Knowledge and Expression

What an AI knows is not always what it says. This gap between internal representation and external behavior has become starkly visible in recent work.

**Rager et al. (2025)** study DeepSeek models, revealing a striking case of censorship mechanics. The models demonstrably possess knowledge about sensitive topics&mdash;their internal representations encode the relevant concepts&mdash;yet they refuse to express this knowledge in their outputs. The information exists inside the model; the suppression is a separate mechanism layered on top.

This dissociation between knowing and saying is not merely a curiosity about Chinese AI policy. It points to a general phenomenon: models may have internal states that diverge systematically from their outputs. If we want to understand what models actually believe, know, or intend, we cannot rely solely on their words. We must look inside.

---

## Causal Mediation: Finding the Neurons That Matter

How do we identify which internal components are responsible for specific behaviors? Early work on generative adversarial networks (GANs) provides an instructive template.

In research on GAN image synthesis, it became possible to identify individual neurons that controlled specific scene properties. A single neuron might control whether lights in a scene are on or off. Activating that neuron turns on the lights; suppressing it turns them off. This is causal mediation analysis: intervening on internal components to establish their causal role in the model's outputs.

The methodology generalizes from GANs to language models. If we can identify the internal representations that causally mediate specific behaviors&mdash;not just correlate with them, but actually cause them&mdash;we gain both scientific understanding and practical control.

---

## Key Research Threads

Three lines of work illustrate the progress that has been made in localizing and characterizing neural concepts.

**Meng et al. (2022)** introduced ROME (Rank-One Model Editing), demonstrating that factual associations in GPT-style models are localized in specific MLP layers. By applying causal tracing&mdash;corrupting inputs and restoring activations at targeted locations&mdash;they identified where facts like "The Eiffel Tower is in Paris" are stored. This work established that factual knowledge has a discernible address inside the network.

**Todd et al. (2024)** extended this localization paradigm from facts to functions. When a model performs in-context learning of an abstract task&mdash;translating to French, answering with antonyms, continuing a pattern&mdash;where is that function encoded? Function vectors show that task-level abstractions are also localizable and, remarkably, transferable: a vector extracted from one context can induce the same function in another.

**Prakash et al. (2025)** examined how models represent and track the mental states of agents described in text&mdash;theory of mind. This work shows that interpretability methods can reach beyond simple factual recall into the representation of genuinely abstract, cognitively rich concepts.

---

## Induction Heads and the Emergence of Language-Independent Concepts

Among the most thoroughly analyzed circuits in transformers are induction heads: attention patterns that implement a simple but powerful operation of matching previous tokens and copying what came next. This mechanism underlies much of in-context learning.

Work by Feucht and colleagues has characterized induction heads in detail, including the Dual Route model that describes how multiple pathways contribute to in-context learning behavior. Crucially, this research reveals that the relevant internal representations are often language-independent. A concept that emerges in English contexts may be the same internal structure that handles the analogous German or Chinese context.

This finding reinforces the central thesis: concepts inside LLMs are not words. They are abstract structures that remain stable under transformations&mdash;switching languages, rephrasing prompts, varying surface details&mdash;that change everything about the tokens while preserving the underlying meaning.

---

## Concepts as Invariants

What, then, is a concept inside a neural network?

The emerging picture suggests that concepts are invariants of the system&mdash;internal structures whose functional roles remain unchanged under many transformations. Just as a physical law remains valid regardless of the coordinate system used to express it, a neural concept may persist across languages, phrasings, and contexts.

Some examples of concepts that appear to have this invariant character:

- **User concepts**: Models appear to maintain representations of who they are talking to, tracking properties of the user across a conversation.
- **Humor concepts**: The recognition that something is funny seems to involve consistent internal structures even as the surface form of jokes varies wildly.
- **Truthfulness concepts**: Directions in activation space that correspond to whether a statement is true or false, independent of topic or phrasing.
- **Task concepts**: The abstract notion of "translate" or "summarize" as distinct from any particular instance of translation or summarization.

These are not isolated neurons but distributed patterns of activation. They are not explicitly labeled in the training data but emerge from the structure of the task. And they are not words&mdash;they are what the words point to.

---

## Toward an Atlas of Neural Concepts

If concepts inside LLMs are real, discoverable, and causally important, then we should map them systematically. The vision motivating this course is the creation of an atlas of neural concepts: a comprehensive characterization of the internal structures that drive model behavior across domains.

Such an atlas would serve multiple purposes:

- **Scientific understanding**: What are the basic building blocks of machine cognition?
- **Safety and alignment**: Which concepts relate to honesty, deception, harm, and helpfulness?
- **Knowledge extraction**: What has the model learned that humans do not yet know?
- **Control and steering**: How can we intervene on concepts to shape behavior?

Building this atlas requires methods. The works surveyed above&mdash;and the techniques you will learn throughout this course&mdash;constitute the toolkit for this cartographic project.

---

## Discussion Questions

1. **What do these works have in common?** Each of the research threads discussed above&mdash;causal tracing, function vectors, theory of mind representations, induction head circuits&mdash;uses a distinct set of methods. What is the shared methodology that connects them? What makes something a "mechanistic interpretability" approach?

2. **What belongs in the atlas?** If we were to construct an atlas of neural concepts, what should it contain? Individual neurons? Directions in activation space? Circuits? Functional roles? What level of description is most useful, and for whom?

3. **How do we know when we have found a concept?** What criteria distinguish a genuine neural concept from a spurious correlation or an artifact of our analysis methods? How do we validate that what we have found is real?

4. **What concepts matter most?** Given limited research resources, which concepts should we prioritize understanding? What makes a concept important from a scientific perspective? From a safety perspective? From a practical application perspective?

5. **Can models describe their own concepts?** If concepts are not words, can language models ever articulate what they internally represent? Or is there a fundamental gap between neural representation and linguistic expression?

---

## References

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *Advances in Neural Information Processing Systems, 35*, 17359&ndash;17372.

Prakash, N., Shaham, T. R., Gross, T., Bau, D., & Belinkov, Y. (2025). Fine-tuning enhances existing mechanisms: A case study on entity tracking. *ICLR 2024*.

Rager, S., et al. (2025). Open-source DeepSeek censorship. Preprint.

Todd, E., Li, M. L., Sharma, A., Mueller, A., Wallace, B. C., & Bau, D. (2024). Function vectors in large language models. *ICLR 2024*.
