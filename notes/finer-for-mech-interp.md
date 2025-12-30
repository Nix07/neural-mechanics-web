# Finding a Good Research Question in Mechanistic Interpretability

## Introduction

Before you invest months of effort into a research project, you should ask: is this the right question? The literature on research methodology offers surprisingly useful guidance here, even though most of it comes from fields far removed from machine learning.

Three essays deserve your attention. Richard Hamming's 1986 talk "You and Your Research" asks why some scientists make lasting contributions while others are forgotten. His central provocation: "If you do not work on an important problem, it's unlikely you'll do important work." Michael Nielsen's "Principles of Effective Research" (2004) distinguishes problem-solvers from problem-creators and argues that most researchers underinvest in the latter. And Cummings, Browner, and Hulley's chapter "Conceiving the Research Question" from *Designing Clinical Research* offers the FINER framework: a good research question should be Feasible, Interesting, Novel, Ethical, and Relevant.

The FINER framework was designed for clinical medicine, but it adapts well to mechanistic interpretability. What follows is my attempt at that adaptation.

A note on the structure of this course: we are forming interdisciplinary teams, pairing PhD students from outside AI with graduate students in machine learning. This is not an accident. In the era of large language models, the greatest societal impacts will ultimately occur outside computer science: in law, medicine, education, science, policy, economics, the humanities. Yet almost no interpretability research examines how models represent concepts from these domains. The lawyer on your team knows what legal reasoning actually requires. The biologist knows which representations would matter for scientific discovery. The economist knows which decision-making processes have consequences. This is your advantage. Use it.

## Feasibility

Can you actually look inside an AI that exhibits the phenomenon you care about?

This question has layers. You need a model that (1) exhibits the behavior, (2) is small enough to study, and (3) actually reveals something when you look inside. Many projects fail at step three: the model exhibits the behavior, but the internal representations are opaque, distributed, or otherwise resistant to interpretation.

A practical workflow:

**Choose a class of phenomena.** What aspect of model behavior interests you? Factual recall, reasoning, safety-relevant behaviors, linguistic competence, multimodal integration? Your choice here reflects your taste and your bets about what matters.

**Narrow down by eliciting the behavior in a frontier model.** Before committing to a phenomenon, verify that you can reliably elicit it. Use ChatGPT, Claude, or Gemini. Can you construct prompts that consistently produce the behavior? If the behavior is fragile or inconsistent even in the best models, interpretability work will be harder.

**Find the smallest open model that reproduces it.** You will spend many hours staring at activations, running interventions, and iterating on hypotheses. This is far easier in a 1B parameter model than a 70B parameter model. The goal is to find the smallest model where the phenomenon still appears. Pythia, Llama, Gemma, and Qwen all offer families of models at multiple scales.

**Look for signs of life.** Here is where feasibility becomes concrete. Run your probes. Compute your activation patching. Look at attention patterns. Do you see internal representations that suggest the model is processing the concept you care about? If the first few hours of exploration reveal nothing: either your tools are wrong, your hypotheses are wrong, or the phenomenon may not be interpretable with current methods.

Signs of life matter. A project with clear internal structure visible from the start will likely yield results. A project where everything looks like noise may never converge.

One more consideration: as your research matures, you will need to show that your findings generalize beyond the toy model where you developed them. Keep an eye on whether the phenomenon appears in multiple models. If your circuit exists only in Pythia-70M and nowhere else, reviewers will question whether you have discovered something general.

## Interesting

Is this question interesting to you and to your field?

The two are not identical. Hamming spent Friday lunches discussing "great thoughts" with colleagues: not the problems they were currently solving, but the problems they believed mattered most. He found that many smart people worked on problems they themselves did not consider important.

Your own interest sustains you through the months of debugging, failed experiments, and rejected papers. But you should also ask: if I solve this, will anyone care? Talk to colleagues. Present your question at lab meetings before you have results. Their reactions reveal whether you are onto something.

Here is where interdisciplinary teams have an advantage. The interpretability community has developed strong intuitions about which questions matter for AI safety and for understanding deep learning. But we have weak intuitions about which questions matter for medicine, for law, for scientific discovery, for education. Your non-CS collaborators have those intuitions. When the legal scholar on your team gets excited about a research direction, that signal contains information you could not generate yourself.

The inverse also holds. If you propose a question and your domain expert shrugs, ask why. Perhaps the phenomenon you find technically fascinating does not correspond to anything practitioners actually care about. Better to learn this in week two than in month eight.

A warning: the most interesting phenomena may be the hardest to study. There is a tradeoff between importance and tractability. Hamming again: "It's not the consequence that makes a problem important, it is that you have a reasonable attack."

## Novelty

There are problems that have been explored many times in the interpretability literature: sentiment classification circuits, subject-verb agreement, numerical representations in small transformers. These were good first targets. They are no longer novel.

Do something new.

The interdisciplinary structure of this course points toward vast unexplored territory. How do language models represent causal reasoning? Legal precedent? Biological mechanism? Historical contingency? Moral uncertainty? These are not idle questions. Models are already being deployed in domains where such representations matter. Yet we have almost no interpretability research examining them.

This is the opportunity. This course will equip you with the interpretability methods: probing classifiers, activation patching, circuit analysis, sparse autoencoders. These techniques will be familiar to the machine learning students on your team. The domain expert contributes something equally essential: knowledge of which concepts matter, how they relate, and what would constitute genuine understanding versus superficial pattern matching. Neither could do this work alone. Together, you can ask questions no one else is asking.

A note of caution: stay open-minded about what you will find. The internal workings of large language models are largely unknown. They may organize their computations in ways that are surprising, or even foreign to the way humans think about these domains. The legal scholar's intuitions about how precedent *should* work may not match how the model actually processes it. This is not a failure; it is a discovery.

Novelty can come from several sources: a new phenomenon, a new method, a new model family, or a new way of connecting interpretability findings to downstream applications. But finding a new phenomenon in an important domain is perhaps the highest-leverage form of novelty available right now.

Nielsen distinguishes problem-solvers from problem-creators. Problem-solvers attack well-posed questions that the community already recognizes as important. Problem-creators identify new questions. Both paths can succeed, but problem-creation has higher variance and higher potential upside. If you find a question no one else has asked, and it turns out to be important, you get the territory to yourself.

## Ethics

Avoid chasing problems that would endanger or disempower people.

Interpretability research is often motivated by safety: we want to understand models so we can make them more reliable, more honest, less prone to catastrophic failures. But interpretability tools can also be misused. A method for identifying deceptive circuits could be inverted to train models that deceive more effectively. A technique for locating safety-relevant representations could be used to ablate them.

Think through the implications of your work before you begin. If your method would primarily enable harm, reconsider.

This is not a call for timidity. Understanding how models work is valuable. But researchers have choices about which capabilities to develop and how to communicate them.

## Relevance

AI is transforming every field. Is the question you are asking relevant to what is actually happening?

Consider where large language models are being deployed: medical diagnosis, legal research, scientific literature review, educational tutoring, policy analysis, financial modeling. The highest-stakes applications are almost entirely outside computer science. Yet the interpretability research community remains focused on questions that matter primarily to machine learning researchers: how do transformers implement algorithms? How do they store factual knowledge? How do safety fine-tuning techniques work?

These are good questions. But they are not the only questions. A doctor deciding whether to trust a model's differential diagnosis needs to know whether the model is reasoning from symptoms to pathophysiology or merely pattern-matching on surface features. A lawyer using a model for case research needs to know whether it understands precedent or is hallucinating citations. These domain-specific questions are at least as important as the CS-centric questions that dominate current research, and they are far less explored.

Relevance is not the same as trendiness. A question can be relevant without appearing in this week's Twitter discourse. But a question can also be irrelevant despite being technically interesting: a deep study of positional encoding schemes that no one uses anymore, or a circuit analysis of a model architecture that has been abandoned.

Ask yourself: if I answer this question completely, what changes? Does it inform how practitioners build or deploy models? Does it advance our theoretical understanding of learning and representation? Does it connect to concerns that people in other fields actually have?

Cummings and colleagues suggest imagining the various possible outcomes of your research and asking whether each would "advance scientific knowledge, influence practice guidelines and health policy, or guide further research." The same test applies here. What would each possible finding mean: not just for machine learning, but for the domains where these models are being used?

## The Iterative Process

Research questions are not found; they are developed. You start with a vague interest, narrow it through reading and exploration, test it against feasibility constraints, refine it through conversation with colleagues, and often abandon it in favor of something better.

Cummings describes this as "an iterative process of making incremental changes in the study's design, estimating the sample size, reviewing with colleagues, pretesting key features, and revising." The interpretability version: make incremental changes in your experimental setup, estimate what compute you need, review with colleagues, run pilot experiments, and revise.

Write down your research question early, even if it feels premature. A one-paragraph description of what you want to learn and why forces clarity. Show it to your advisor. Show it to other students. Their confusions reveal where your thinking is still muddled.

## Summary

Before committing to a research project, ask:

- **Feasible:** Can I find a model where the phenomenon appears and where I can see inside? Are there signs of life in the internal representations?
- **Interesting:** Does this question excite me? Would my colleagues consider it important?
- **Novel:** Has this been done before? What is new about my approach?
- **Ethical:** Would this work primarily enable harm?
- **Relevant:** If I answer this question, what changes for the field?

These criteria are not a checklist to satisfy once and forget. Return to them throughout your project. The best research often involves discovering, midstream, that you were asking the wrong question: and having the flexibility to ask a better one.

## References

Cummings, S. R., Browner, W. S., & Hulley, S. B. (2013). Conceiving the research question and developing the study plan. In S. B. Hulley et al. (Eds.), *Designing Clinical Research* (4th ed., pp. 14–22). Lippincott Williams & Wilkins.

Hamming, R. W. (1986, March 7). You and your research [Colloquium presentation]. Bell Communications Research, Morristown, NJ. Transcript available at cs.virginia.edu/~robins/YouAndYourResearch.html

Nielsen, M. A. (2004). Principles of effective research (Technical Note 0404). University of Queensland.
