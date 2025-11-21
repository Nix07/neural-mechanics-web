# Course Structure: Alternatives and Rationale

## Current Structure (Before Reorganization)

| Week | Topic |
|------|-------|
| 0 | Introduction & Course Overview |
| 1 | Benchmarking |
| 2 | Steering |
| 3 | Representation Visualization |
| 4 | Causal Mediation Analysis |
| 5 | Circuits and Mechanistic Analysis |
| 6 | Probes and Masks |
| 7 | Unsupervised Feature Discovery (SAEs) |
| 8 | Causal Abstraction and Validation |
| 9 | Input Attribution and Saliency Methods |
| 10 | Skepticism and Interpretability Illusions |
| 11 | Bridging the Human-AI Knowledge Gap |
| 12 | Project Presentations |

---

## Problems Identified

### 1. **Skepticism Comes Too Late (Week 10)**

**Issue:** Students form project hypotheses and run experiments in Weeks 1-9 without critical validation frameworks. By Week 10, they may have committed to flawed interpretations.

**Evidence:** Week 10 teaches that:
- Many saliency methods fail sanity checks (Adebayo)
- Attention weights don't explain predictions (Jain & Wallace)
- Most feature importance methods perform no better than random (Hooker)
- Circuit faithfulness depends on methodology (Miller)
- Neurons appear monosemantic but aren't (Bolukbasi)

**Impact:** HIGH - Students need this before forming project interpretations.

### 2. **Attribution Methods Too Late (Week 9)**

**Issue:** Input attribution (gradients, IG, LIME, SHAP) is taught after circuits (Week 5) which uses attribution internally, and after probes (Week 6) which could benefit from attribution.

**Impact:** MEDIUM - Creates dependency issues, students may want attribution earlier for projects.

### 3. **Method-Critique Mismatch**

**Issue:** If skepticism is moved earlier (e.g., Week 5), it critiques methods not yet taught:
- Adebayo critiques saliency maps (not taught until Week 9)
- Jain & Wallace critique attention visualization
- Hooker critiques feature importance methods
- Bolukbasi critiques neuron interpretations
- Miller critiques circuit faithfulness metrics (circuits taught Week 5)

**Impact:** HIGH - Pedagogically backwards to critique before teaching.

### 4. **Interventional Methods Scattered**

**Issue:** Steering (Week 2) and Causal Mediation (Week 4) are interventional "gold standard" methods separated by observational methods. Students don't get clear sense of intervention vs. observation distinction.

**Impact:** MEDIUM - Conceptual clarity issue.

### 5. **Circuits and Causal Abstraction Separated**

**Issue:** Circuits (Week 5) and Causal Abstraction (Week 8) are naturally paired - causal abstraction provides the principled framework for validating circuits. Separating them means students learn circuits with ad-hoc validation, then later learn the right way.

**Impact:** MEDIUM - Students learn problematic metrics first.

---

## Proposed Solution: Integrated Skepticism with Sandwich Structure

### Core Principles

1. **Interventional methods taught early** - Establish causal validation as the gold standard
2. **Skepticism integrated with methods** - Critique each method immediately after teaching it
3. **General validation framework** - Week 4 teaches principles applicable to all methods
4. **Circuits + Causal Abstraction combined** - Teach principled validation from the start
5. **Observational methods framed as hypothesis generation** - Must be validated with interventions

### Structure Overview

**Phase 1: Interventional Bootcamp (Weeks 1-4)**
- Establish causal validation as foundation
- Teach general validation principles

**Phase 2: Observational Methods with Built-in Critique (Weeks 5-7)**
- Each method taught with its critique integrated
- Constant reminder to validate with interventions

**Phase 3: Synthesis (Weeks 8-9)**
- Circuits + Causal Abstraction combined
- Human validation studies

**Phase 4: Project Work (Weeks 10-12)**

---

## Detailed Week-by-Week Structure

### **Week 0: Introduction & Course Overview**
*No change from current structure*

---

### **Week 1: Benchmarking**
*No change from current structure*

**Content:**
- What does good interpretability look like?
- Evaluation metrics
- Case studies of successful interpretability research

---

### **Week 2: Steering**
*No change from current structure*

**Content:**
- Activation addition/subtraction
- Representation engineering
- Immediate behavioral feedback
- Simple, concrete interventions

**Why Week 2:**
- Pedagogically easier than causal mediation (more concrete)
- Students see immediate results
- Introduces intervention concept simply

---

### **Week 3: Representation Visualization**
*No change from current structure*

**Content:**
- PCA, t-SNE, UMAP
- Activation atlases
- Neuroscope-style visualization
- Understanding what activations are

**Why Week 3:**
- Students need to understand activations before patching them (Week 4)
- Provides intuition for later methods
- Simple observational method

---

### **Week 4: Causal Mediation Analysis + Validation Framework**
*MODIFIED - Added validation framework component*

#### **Part 1 (60%): Causal Mediation Analysis**

**Papers:**
1. Vig et al. (2020) - "Investigating Gender Bias in Language Models Using Causal Mediation Analysis"
2. Meng et al. (2022) - "Locating and Editing Factual Associations in GPT" (ROME)

**Content:**
- Activation patching
- Causal tracing
- Testing information flow
- Counterfactual interventions

#### **Part 2 (40%): General Validation Framework**

**Papers:**
3. Doshi-Velez & Kim (2017) - "Towards A Rigorous Science of Interpretable Machine Learning"
4. Jacovi & Goldberg (2020) - "Towards Faithfully Interpretable NLP Systems"

**Framework (applied throughout Weeks 5-9):**
1. **Multi-method validation** - Always use ≥3 independent techniques
2. **Causal validation** - Use interventions to validate observational findings
3. **Sanity checks** - Test on random model/labels
4. **Baseline comparisons** - Compare to random, simple heuristics
5. **Counterfactual testing** - Minimal input edits

**Exercise:** Students implement causal tracing and apply validation framework to test a simple interpretation.

**Why Week 4:**
- Establishes gold standard before observational methods
- Students understand activations (Week 3) before patching them
- Validation principles apply to all subsequent weeks

---

### **Week 5: Probes and Masks (+ TCAV/CBMs)**
*MOVED from Week 6, ADDED TCAV/CBMs*

**Content:**

#### **Part 1: Linear Probes**
- Training classifiers on activations
- What's linearly represented?
- Probe design choices

#### **Part 2: TCAV and Concept Bottleneck Models**
- Testing with Concept Activation Vectors
- Concept-based explanations
- Directional derivatives

#### **Part 3: Validation**
- Apply Week 4's framework
- Validate probe findings with causal interventions
- When do probes fail?

**Why Week 5:**
- Observational method (hypothesis generation)
- Builds on understanding of activations (Week 3)
- Students can validate with causal mediation (Week 4)
- No method-specific critique papers (so no mismatch)

---

### **Week 6: Input Attribution and Saliency Methods + Skepticism**
*MOVED from Week 9, INTEGRATED skepticism papers*

**Content:**

#### **Part 1 (50%): Attribution Methods**
- Gradient-based: Saliency, Input×Gradient, Integrated Gradients, DeepLIFT
- Perturbation-based: LIME, SHAP, ablation
- Attention-based: Attention rollout, attention flow
- Inseq library

#### **Part 2 (50%): When Attribution Fails**

**Papers:**
1. **Adebayo et al. (2018) - "Sanity Checks for Saliency Maps"**
   - Guided Backprop fails sanity checks
   - Model randomization test
   - Data randomization test

2. **Hooker et al. (2019) - "A Benchmark for Interpretability Methods in Deep Neural Networks" (ROAR)**
   - Most methods no better than random
   - RemOve And Retrain benchmark

3. **Jain & Wallace (2019) - "Attention is not Explanation"**
   - Attention weights uncorrelated with importance
   - Multiple attention patterns yield same output

**Additional Topics:**
- LIME/SHAP additivity violations (transformers aren't additive)
- Gradient saturation problems
- Baseline selection for IG

**Exercise:** Students implement saliency maps AND run Adebayo's sanity checks immediately.

**Why Week 6:**
- NOW students can understand the critiques (they just learned attribution)
- Critical thinking before generating project hypotheses
- Natural pairing: method + critique

---

### **Week 7: Unsupervised Feature Discovery (SAEs) + Interpretability Illusions**
*MOVED from Week 8, INTEGRATED skepticism papers*

**Content:**

#### **Part 1 (50%): Sparse Autoencoders**
- Dictionary learning
- Monosemantic features
- Anthropic's work (Golden Gate Bridge example)
- SAE training and interpretation

#### **Part 2 (50%): The Illusions**

**Papers:**
1. **Bolukbasi et al. (2021) - "An Interpretability Illusion for BERT"**
   - Individual neurons appear monosemantic but aren't
   - Dataset-level vs. global concepts
   - Geometric properties of embedding space
   - Counterfactual testing reveals failures

2. **"Interpretability Illusions with Sparse Autoencoders" (2025)**
   - SAE features are adversarially fragile
   - Tiny perturbations change activations without changing outputs
   - Robustness as validation criterion

**Framework:**
- Local vs. global vs. dataset-level concepts
- Testing generalization to counterfactuals
- Adversarial robustness testing

**Exercise:** Students extract SAE features AND test robustness with counterfactuals.

**Why Week 7:**
- Students understand SAEs before seeing their limitations
- Natural pairing: unsupervised discovery + why it can be illusory
- Reinforces Week 4's counterfactual testing principle

---

### **Week 8: Circuits + Causal Abstraction**
*COMBINED Weeks 5 and 8*

**Content:**

#### **Part 1 (30%): Mechanistic Interpretability**
- What are circuits?
- Finding computational subgraphs
- Examples: IOI, greater-than, factual recall
- Identifying candidate circuits

#### **Part 2 (40%): Causal Abstraction Framework**
- Formal validation theory (Geiger et al.)
- Causal scrubbing - the principled approach
- Interchange interventions
- Why this is better than ad-hoc ablation metrics

**Papers:**
1. Wang et al. (2022) - "Interpretability in the Wild: a Circuit for Indirect Object Identification"
2. Geiger et al. (2021) - "Causal Abstractions of Neural Networks"
3. Geiger et al. (2023) - "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations"

#### **Part 3 (30%): Validating Your Circuits**
- Apply causal abstraction to circuit validation
- Practical implementation
- Integration with Weeks 2-4 intervention techniques

**Brief mention:**
- Miller et al. (2024) - "Transformer Circuit Faithfulness Metrics are not Robust"
- Why ad-hoc ablation metrics fail (justifies causal abstraction approach)

**Why Week 8:**
- Circuits + validation taught together (principled from start)
- Students already know interventions (Weeks 2, 4)
- De-emphasizes problematic faithfulness metrics
- Causal abstraction as THE right way

---

### **Week 9: Human Validation Studies + Knowledge Gap**
*COMBINED elements from Weeks 10-11*

**Content:**

#### **Part 1 (40%): Do Explanations Help Humans?**

**Papers:**
1. **Bansal et al. (2021) - "Does the Whole Exceed its Parts?"**
   - Explanations increase trust but not accuracy
   - Humans accept AI recommendations indiscriminately
   - Cognitive offloading and confirmation bias

**Framework:**
- Three types of human evaluation (Doshi-Velez from Week 4)
- Designing human studies
- Measuring complementary team performance

#### **Part 2 (60%): Bridging the Human-AI Knowledge Gap**

**Papers:**
2. **Shin et al. (2023)** - Learning from AlphaGo (passive)
3. **Schut et al. (2023/2025)** - Teaching grandmasters (active)

**Framework:**
- Teachability as validation criterion
- Novelty assessment
- Extracting superhuman knowledge
- Human validation study design

**Exercise:** Students design a human validation study for their project concept.

**Why Week 9:**
- Synthesis week - combines validation with knowledge extraction
- Students have all tools (Weeks 1-8) to interpret concepts
- Now test if interpretations help humans
- Perfect timing for project validation

---

### **Weeks 10-12: Project Work and Presentations**

**Week 10-11:** Project workshops
- Students present progress
- Peer validation and red-teaming
- Apply Weeks 4-9 frameworks to projects
- Office hours and individual guidance

**Week 12:** Final presentations

---

## Rationale for Proposed Structure

### 1. **Interventional Bootcamp (Weeks 1-4)**

**Rationale:** Establish causal validation as the gold standard before students learn observational methods.

**Benefits:**
- Students understand that intervention > observation for causal claims
- Validation framework (Week 4) applies to all subsequent methods
- Steering (Week 2) before causal mediation (Week 4) is pedagogically sound: simple → complex

**Key Insight:** Students need to understand activations (Week 3) before patching them (Week 4).

### 2. **Integrated Skepticism (Weeks 5-9)**

**Rationale:** Critique each method immediately after teaching it - no method-critique mismatch.

**Benefits:**
- Students understand what's being critiqued (just learned it)
- Immediate application in exercises (implement method AND run validation)
- Builds critical thinking gradually, not as overwhelming "skepticism dump"
- Natural flow: learn → critique → validate

**Key Insight:** You can't critique saliency maps (Adebayo) before students know what saliency maps are.

### 3. **Circuits + Causal Abstraction Combined (Week 8)**

**Rationale:** Teach principled validation (causal abstraction) from the start, not problematic ad-hoc metrics.

**Benefits:**
- Students learn the RIGHT way immediately
- Causal abstraction provides theoretical foundation
- De-emphasizes faithfulness metrics (which Miller showed are problematic)
- Natural pairing: circuits (what to validate) + causal abstraction (how to validate)

**Key Insight:** Teaching circuits with ablation metrics, then later teaching causal abstraction, means students learn the wrong way first.

### 4. **Observational Methods as Hypothesis Generation (Weeks 5-7)**

**Rationale:** Frame probes, attribution, SAEs as fast exploration tools that MUST be validated.

**Benefits:**
- Students understand these are correlational, not causal
- Constant reminder to validate with Week 4's interventions
- Appropriate epistemic humility about observational findings

**Key Insight:** Observational methods are useful for forming hypotheses, interventional methods test them.

### 5. **Human Validation Synthesis (Week 9)**

**Rationale:** After learning all technical methods, test if interpretations actually help humans.

**Benefits:**
- Students have complete technical toolkit (Weeks 1-8)
- Human validation as ultimate test
- Perfect timing for project design
- Connects interpretability to real-world impact

---

## Alternative Approaches Considered

### **Alternative 1: Original Structure with Skepticism Moved Earlier**

**Structure:**
- Weeks 1-3: Same
- **Week 4: Skepticism Fundamentals**
- Week 5: Causal Mediation
- Weeks 6-9: Other methods
- Week 10: Advanced Skepticism
- Week 11: Knowledge Gap

**Pros:**
- Early critical thinking
- Simple change from current structure

**Cons:**
- **Method-critique mismatch**: Week 4 critiques saliency (taught Week 9), attribution (Week 9), attention, circuits (Week 5)
- Students read papers about methods they haven't learned
- Pedagogically backwards

**Verdict:** REJECTED due to method-critique mismatch.

---

### **Alternative 2: Two Skepticism Weeks**

**Structure:**
- Week 4: General skepticism principles (method-agnostic)
- Weeks 5-9: Methods
- Week 10: Method-specific critiques (Adebayo, Jain & Wallace, Hooker, Miller, Bolukbasi)

**Pros:**
- Week 4 avoids method-critique mismatch by staying general
- Students get early warning
- Week 10 provides detailed critiques after all methods taught

**Cons:**
- Week 4 would be abstract without concrete examples
- Students might not retain general principles without application
- Still gap between learning method (e.g., Week 6 attribution) and critique (Week 10)

**Verdict:** REJECTED - too much abstraction in Week 4, critique still delayed.

---

### **Alternative 3: Observational Methods First, Then Interventional**

**Structure:**
- Week 1: Benchmarking
- Weeks 2-5: Observational methods (Visualization, Probes, Attribution, SAEs)
- Weeks 6-8: Interventional methods (Steering, Causal Mediation, Circuits)
- Week 9: Skepticism
- Week 10: Knowledge Gap

**Pros:**
- Simple to complex progression
- Group similar method types

**Cons:**
- **Students form hypotheses without validation tools** - This is backwards scientifically
- Learn correlation before causation
- Doesn't emphasize intervention as gold standard
- Students might commit to flawed interpretations early

**Verdict:** REJECTED - scientifically backwards to teach observation before intervention.

---

### **Alternative 4: Pair Each Observational Method with Validation**

**Structure:**
- Week 2: Steering
- Week 3: **Causal Mediation + Probes** (paired)
- Week 4: **Visualization + Attribution** (paired)
- Week 5: **SAEs + Validation**
- Week 6: Circuits + Causal Abstraction
- Week 7: Skepticism

**Pros:**
- Immediate connection between hypothesis and test
- No gap between method and validation

**Cons:**
- Weeks become very dense (two major topics each)
- Might overwhelm students
- Hard to teach causal mediation AND probes in one week
- Loses pedagogical build-up

**Verdict:** REJECTED - too dense, pedagogically challenging.

---

### **Alternative 5: Minimal Changes**

**Structure:**
- Keep current order (Weeks 1-9)
- Add brief "Critical Thinking Preview" to Week 1
- Move attribution from Week 9 to Week 7
- Keep skepticism Week 10 but frame as "revisiting with advanced topics"

**Pros:**
- Minimal disruption
- Familiar structure
- Easy to implement

**Cons:**
- **Doesn't solve main problem**: Skepticism still too late
- Students still form interpretations Weeks 1-9 without critical lens
- Only addresses attribution placement, not core issue

**Verdict:** REJECTED - doesn't address main problem.

---

### **Alternative 6: Standalone Skepticism After Methods**

**Structure:**
- Weeks 1-9: All methods (current order, with light validation mentions)
- **Week 10: Deep Dive Skepticism** - All critique papers (Adebayo, Jain & Wallace, Hooker, Miller, Bolukbasi, SAE illusions, Bansal)
- Week 11: Knowledge Gap

**Pros:**
- Unified critical perspective
- No method-critique mismatch (all methods taught first)
- Comprehensive skepticism treatment

**Cons:**
- **Students already formed interpretations** by Weeks 5-9
- May have committed to flawed project approaches
- Skepticism comes too late to change behavior
- Might feel like "gotcha" - "everything you learned is wrong"

**Verdict:** REJECTED - core problem persists (skepticism too late).

---

## Comparison Table

| Approach | Method-Critique Mismatch | Skepticism Timing | Intervention Emphasis | Pedagogical Flow | Implementation Difficulty |
|----------|-------------------------|-------------------|---------------------|------------------|-------------------------|
| **Current** | None (but skepticism late) | Week 10 (too late) | Medium | Good | N/A (baseline) |
| **Proposed (Integrated)** | None | Throughout (Weeks 4-9) | High | Excellent | Medium |
| Alt 1: Early Skepticism | **HIGH** | Week 4 | Medium | Poor | Low |
| Alt 2: Two Skepticism Weeks | Medium (Week 4 abstract) | Weeks 4 & 10 | Medium | Medium | Medium |
| Alt 3: Observational First | None | Week 9 | Low | Poor | Low |
| Alt 4: Paired Methods | None | Throughout | High | Challenging | High |
| Alt 5: Minimal Changes | None | Week 10 (too late) | Medium | Good | Low |
| Alt 6: Standalone After | None | Week 10 (too late) | Medium | Good | Low |

**Legend:**
- **Method-Critique Mismatch:** Does skepticism critique methods not yet taught?
- **Skepticism Timing:** When do students learn critical thinking?
- **Intervention Emphasis:** How strongly emphasized are causal interventions?
- **Pedagogical Flow:** How natural is the learning progression?

---

## Implementation Considerations

### **Major Changes from Current Structure**

1. **Week 4:** Add validation framework (Part 2) to causal mediation
2. **Week 5:** Move probes from Week 6, add TCAV/CBMs
3. **Week 6:** Move attribution from Week 9, add Adebayo/Hooker/Jain & Wallace
4. **Week 7:** Add Bolukbasi and SAE illusions to SAE week
5. **Week 8:** Combine circuits (Week 5) and causal abstraction (Week 8)
6. **Week 9:** Combine human studies and knowledge gap

### **Content Shifts**

**Weeks gaining content:**
- Week 4: +40% (validation framework)
- Week 6: +50% (skepticism papers)
- Week 7: +50% (illusions papers)

**Weeks losing content:**
- Week 10: Now project workshop (lost all content to earlier weeks)
- Week 11: Now project workshop (lost knowledge gap to Week 9)

### **Exercise Modifications**

Every Week 5-9 exercise must now include validation component:
- Week 5 (Probes): Validate probe findings with causal interventions
- Week 6 (Attribution): Run Adebayo sanity checks on saliency maps
- Week 7 (SAEs): Test SAE features with counterfactuals and adversarial robustness
- Week 8 (Circuits): Use causal scrubbing instead of ablation metrics
- Week 9 (Human): Design human validation study

### **Lecture Time Allocation**

Weeks 6-8 are content-heavy:
- **Week 6:** 1.5 hours method + 1.5 hours critique (feasible)
- **Week 7:** 1.5 hours SAEs + 1.5 hours illusions (feasible)
- **Week 8:** 1 hour circuits + 1.5 hours causal abstraction + 0.5 hours integration (tight but feasible)

Consider:
- Assign papers as pre-reading
- Focus lecture on synthesis and key insights
- Move some technical details to exercises

---

## Recommendation

**Implement the Proposed Integrated Skepticism Structure.**

**Why:**

1. **Solves the core problem:** Skepticism integrated throughout, not delayed to Week 10
2. **No method-critique mismatch:** Students understand methods before critiques
3. **Emphasizes intervention as gold standard:** Clear distinction between observation and causation
4. **Natural pedagogical flow:** Simple → complex, with validation at each step
5. **Project-ready by Week 9:** Students have all technical and critical thinking tools

**Trade-offs accepted:**
- More content-dense weeks (6-8)
- Requires more preparation time for integrated lectures
- Students get less "breathing room" - constant critical evaluation

**Benefits outweigh costs:** Students will produce more rigorous research, form better habits, and avoid common pitfalls.

---

## Next Steps

If approved, implementation requires:

1. **Week 4 materials:**
   - Add validation framework lecture (Part 2)
   - Add Doshi-Velez and Jacovi papers
   - Update exercises to include validation

2. **Week 6 materials:**
   - Move attribution content from Week 9
   - Add Adebayo, Hooker, Jain & Wallace papers
   - Update exercises to include sanity checks

3. **Week 7 materials:**
   - Add Bolukbasi and SAE illusions papers
   - Update exercises to include counterfactual testing

4. **Week 8 materials:**
   - Combine circuits and causal abstraction
   - Restructure to emphasize causal abstraction as foundation
   - Update exercises to use causal scrubbing

5. **Week 9 materials:**
   - Combine human studies and knowledge gap
   - Add exercise on study design

6. **Week 10-11:**
   - Repurpose as project workshops
   - Create workshop format and guidance materials

7. **Update index.html:**
   - Update week titles and descriptions
   - Update all cross-references between weeks

---

## Appendix: Week 4 Papers - Detailed Options

### **Option A: Methodological Principles (No Additional Papers)**

Week 4 Part 2 teaches principles without dedicated papers, applied in later weeks.

**Pros:** Practical, actionable, no extra reading burden
**Cons:** Less grounded in literature

### **Option B: Include Meta-Papers**

**Papers:**
- Doshi-Velez & Kim (2017)
- Jacovi & Goldberg (2020)
- Optional: Lipton (2018)

**Pros:** Conceptual foundation, highly cited frameworks
**Cons:** More abstract, less immediately practical

### **Option C: Hybrid (Recommended)**

**Papers:** Doshi-Velez + Jacovi
**Lecture:** Practical principles (multi-method, causal, sanity checks, baselines, counterfactuals)

**Rationale:** Best of both - conceptual foundation + practical application

---

*Document Version: 1.0*
*Last Updated: 2025-01-21*
