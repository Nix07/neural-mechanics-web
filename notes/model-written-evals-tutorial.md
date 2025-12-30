# Model-Written Evaluations for Interpretability Research

A tutorial based on Perez et al. (2022), "Discovering Language Model Behaviors with Model-Written Evaluations"

---

## Overview

Before we can study how a concept is represented inside a neural network, we need examples that reliably elicit that concept. Creating evaluation datasets by hand is slow. Perez et al. demonstrate that language models can generate evaluation data at scale—and that this approach can uncover behaviors humans might not anticipate.

This tutorial explains the core methodology and adapts it for interpretability research, using **pun understanding** as a running example.

---

## The Core Method

The paper's approach has four stages:

1. **Specify the behavior** you want to evaluate in a clear prompt
2. **Generate examples** by prompting a capable LLM with instructions and a few seed examples
3. **Filter for quality** using automated checks and human review
4. **Run evaluations** on target models and analyze results

The key insight: LLMs can generate diverse, creative test cases faster than humans, while humans remain in the loop for quality control and analysis.

---

## Evaluation Formats

The paper uses several evaluation formats, each with different strengths.

### Why Token Localization Matters for Interpretability

Many interpretability methods become dramatically easier when the model's critical decision is concentrated at a single, predictable token position. Consider what you can do when you know exactly where the decision happens:

- **Logit lens**: Decode intermediate layers at the decision token to watch the prediction evolve
- **Activation patching**: Intervene at a known position rather than searching across the sequence
- **Probing**: Extract activations at exactly the decision point, avoiding noise from irrelevant positions
- **Attention analysis**: See what the model attends to when generating the critical token
- **Probability comparison**: Compare logits for candidate completions directly

Two formats naturally provide this localization: **cloze-style prompts** (fill-in-the-blank) and **MCQ** (where the model produces a single letter). Free-form generation is harder to analyze because the "decision" is distributed across multiple tokens.

### Cloze-Style Evaluation

The model completes a sentence with a single word or short phrase. The blank is the decision point.

**Strengths**: Decision localized to one token. Enables direct probability comparison over candidate completions. Natural fit for logit lens and activation extraction.

**Weaknesses**: Requires careful prompt design so the blank is unambiguous. Some concepts do not reduce naturally to single-token decisions.

**Design principle**: The cloze prompt should be constructed so that (1) the correct answer is a single token or very short phrase, and (2) incorrect answers are also plausible completions. This lets you compare P(correct) vs. P(incorrect) at the decision position.

### Multiple Choice Questions (MCQ)

The model chooses among labeled options (A, B, C, D). This enables precise measurement and easy automation.

**Strengths**: Decision localized to a single token (the letter). Unambiguous evaluation. Enables comparison of answer probabilities across options. Easy to automate.

**Weaknesses**: May not reflect real-world usage. The correct answer might be guessable from surface features. Format itself may activate "test-taking" behaviors distinct from natural understanding.

**For interpretability**: MCQ is excellent because you can extract activations at the final position and examine the logit distribution over {A, B, C, D}. You can also study what information flows to that position from earlier in the prompt.

### Zero-Shot Evaluation

The model receives only a question or prompt, with no examples. This tests the model's baseline capabilities and default behaviors.

**Strengths**: Clean signal about what the model "knows" without scaffolding. Good for probing default representations.

**Weaknesses**: May underestimate capability if the model needs context to understand the task format. Free-form responses are harder to analyze.

**For interpretability**: Combine zero-shot with cloze format. Rather than asking "Is this a pun?" and parsing a free-form answer, use "This is a pun: Yes or No? Answer: ___" so the decision is localized.

### In-Context Learning (ICL) Evaluation

The model receives several examples before the test case. This activates task-relevant circuits and can elicit capabilities the model has but does not spontaneously display.

**Strengths**: Tests whether the model *can* perform a task when given appropriate context. Useful for studying how in-context examples modulate internal representations.

**Weaknesses**: Performance depends heavily on example selection. The model may be pattern-matching rather than understanding.

**For interpretability**: ICL is valuable precisely because it changes internal representations. You can compare activations with and without in-context examples to see how demonstrations reshape the model's processing. Use cloze or MCQ format for the test case to maintain token localization.

---

## Case Study: Evaluating Pun Understanding

Suppose we want to study how language models represent **puns**—wordplay that exploits multiple meanings or similar sounds. Before we can localize "pun circuits" or probe for pun-related features, we need evaluation data that reliably triggers pun processing.

### Step 1: Specify the Behavior

We want to evaluate whether a model:
- Recognizes that a statement is a pun
- Understands *why* it is a pun (which words have double meanings)
- Can distinguish good puns from bad ones
- Can generate puns (though this is harder to evaluate)

### Step 2: Generate Examples

We prompt a capable model to generate evaluation data. Here are templates for each format:

#### Cloze-Style Generation Prompt

Cloze prompts localize the decision to a single token, making them ideal for interpretability work. Design the blank so that correct and incorrect answers are both plausible single tokens.

```
Generate cloze-style questions that test pun understanding. Each question 
should have a blank that can be filled with a single word or short phrase.
Provide the correct answer and one plausible incorrect answer.

Format:
Prompt: [sentence with ___ blank]
Correct: [answer]
Incorrect: [plausible wrong answer]

Examples:

Prompt: The joke "I used to be a banker but I lost interest" is a pun 
because the word "interest" refers to both financial returns and ___.
Correct: curiosity
Incorrect: money

Prompt: "Time flies like an arrow; fruit flies like a banana" is a pun 
because "flies" shifts from being a ___ to being a noun.
Correct: verb
Incorrect: metaphor

Prompt: The statement "I'm reading a book on anti-gravity, it's impossible 
to put down" is ___. (Answer: pun / not a pun)
Correct: pun
Incorrect: not a pun

Prompt: In the sentence "The bicycle couldn't stand on its own because it 
was two-tired," the wordplay involves "two-tired" sounding like ___.
Correct: too tired
Incorrect: retired

Generate 20 more cloze-style pun evaluation items with varied formats:
- Some testing pun recognition (is this a pun: yes/no)
- Some testing explanation (what word has double meaning)
- Some testing mechanism (homophone/polysemy/syntactic)
```

**Key design principles for cloze prompts:**
- The correct answer should be a single token when possible (or 2-3 tokens maximum)
- The incorrect answer should be a plausible completion, not obviously wrong
- The blank position should be at the end of the prompt for easiest activation extraction
- Include the answer choices in the prompt when doing forced-choice (e.g., "Answer with Yes or No: ___")

#### Zero-Shot Cloze Generation Prompt

```
Generate pun recognition tasks in cloze format. The model must complete 
the sentence with "Yes" or "No".

Format:
Prompt: [statement]. This is a pun. Yes or No? Answer:
Label: [Yes/No]

Examples:

Prompt: "I used to be a banker, but I lost interest." This is a pun. 
Yes or No? Answer:
Label: Yes

Prompt: "I used to be a banker, but I changed careers." This is a pun. 
Yes or No? Answer:
Label: No

Prompt: "The bicycle couldn't stand because it was two-tired." This is 
a pun. Yes or No? Answer:
Label: Yes

Generate 30 examples, balanced between puns and non-puns. For non-puns, 
use sentences that are structurally similar to puns but lack wordplay:
```

#### MCQ Generation Prompt

```
Generate multiple choice questions that test understanding of puns.
Each question should have one correct answer and three plausible distractors.

Format:
Question: [question text]
A) [option]
B) [option]  
C) [option]
D) [option]
Correct: [letter]

Example:

Question: In the pun "I used to be a banker, but I lost interest," 
which word carries the double meaning?
A) banker
B) lost
C) interest
D) used
Correct: C

Question: Why is "Time flies like an arrow; fruit flies like a banana" 
considered a pun?
A) It compares time to fruit
B) "Flies" shifts from verb to noun, and "like" shifts from comparison to preference
C) Arrows and bananas are both long and thin
D) It rhymes
Correct: B

Generate 15 more MCQ items testing pun recognition and explanation:
```

#### ICL Evaluation Generation Prompt

```
Generate in-context learning evaluation sets for pun understanding. 
Each set should have 3-4 demonstration examples followed by a test case.
The demonstrations should show the format: identify whether something 
is a pun, and if so, explain the wordplay.

Example evaluation set:

[Demonstrations]
Q: "I'm reading a book about anti-gravity. It's impossible to put down."
A: This is a pun. "Put down" means both "stop reading" and "place downward." 
   The joke conflates these meanings.

Q: "I'm reading a book about physics. It's very interesting."
A: This is not a pun. The sentence is straightforward with no wordplay.

Q: "The bicycle couldn't stand on its own because it was two-tired."
A: This is a pun. "Two-tired" sounds like "too tired," creating wordplay 
   between having two tires and being exhausted.

[Test case]
Q: "I wondered why the baseball was getting bigger. Then it hit me."
A: 

---

Generate 10 such evaluation sets with varying difficulty:
```

#### MCQ Generation Prompt

```
Generate multiple choice questions that test understanding of puns.
Each question should have one correct answer and three plausible distractors.

Format:
Question: [question text]
A) [option]
B) [option]  
C) [option]
D) [option]
Correct: [letter]

Example:

Question: In the pun "I used to be a banker, but I lost interest," 
which word carries the double meaning?
A) banker
B) lost
C) interest
D) used
Correct: C

Question: Why is "Time flies like an arrow; fruit flies like a banana" 
considered a pun?
A) It compares time to fruit
B) "Flies" shifts from verb to noun, and "like" shifts from comparison to preference
C) Arrows and bananas are both long and thin
D) It rhymes
Correct: B

Generate 15 more MCQ items testing pun recognition and explanation:
```

### Step 3: Filter for Quality

The paper emphasizes that raw LLM output requires filtering. Apply these checks:

**Automated filters:**
- Remove duplicates and near-duplicates
- Check that MCQ options are distinct
- Verify format compliance (correct labels, expected structure)
- For puns specifically: check that the "pun" examples actually contain wordplay (you might use a second LLM call to verify)

**Human review:**
- Sample 50-100 examples for manual inspection
- Check that puns are actually funny or at least recognizable as wordplay
- Verify that explanations correctly identify the double meaning
- Ensure distractors in MCQs are plausible but clearly wrong

**Diversity checks:**
- Are puns drawn from varied domains (professions, animals, food, etc.)?
- Do they use different mechanisms (homophones, polysemy, syntactic ambiguity)?
- Is difficulty varied?

### Step 4: Run Evaluations

With filtered data, evaluate your target models:

```python
# Pseudocode for MCQ evaluation
def evaluate_mcq(model, questions):
    results = []
    for q in questions:
        # Get log probabilities for each option
        logprobs = model.get_option_logprobs(q.prompt, q.options)
        predicted = argmax(logprobs)
        results.append({
            'correct': predicted == q.answer,
            'confidence': softmax(logprobs)[predicted],
            'logprobs': logprobs
        })
    return results
```

For interpretability work, you likely want more than accuracy:
- **Activation extraction**: Save hidden states when the model processes puns vs. non-puns
- **Attention patterns**: Where does the model attend when processing the ambiguous word?
- **Layer-by-layer analysis**: At which layer does the model "get" the joke?

---

## Pitfalls and How to Avoid Them

The paper identifies several failure modes:

### 1. Generated Examples May Be Low Quality

**Problem**: LLMs sometimes generate examples that are ambiguous, incorrect, or too easy.

**Solution**: Always manually review a sample. For puns, check that the wordplay actually works—LLMs sometimes generate "puns" that are not funny because the double meaning does not quite land.

### 2. Lack of Diversity

**Problem**: LLMs tend to repeat patterns. You might get 50 puns about "bank" and "interest."

**Solution**: 
- Explicitly prompt for diversity: "Generate puns about different topics: food, animals, professions, science, sports..."
- Use multiple generation runs with different seeds or temperatures
- Filter for diversity post-hoc

### 3. Memorization and Data Contamination

**Problem**: The model being evaluated may have seen these exact puns during training. If so, you are testing recall, not understanding.

**Solution**:
- Generate novel puns rather than using famous ones
- Include a novelty check: can the model explain puns it has likely never seen?
- Compare performance on "classic" puns vs. newly generated ones

### 4. Surface Feature Shortcuts

**Problem**: Models might detect puns from surface features (sentence length, punctuation patterns) rather than understanding wordplay.

**Solution**:
- Ensure non-pun examples have similar surface features
- Include "almost-puns" that have the structure but lack the double meaning
- Analyze errors: are failures random or systematic?

### 5. Evaluation Format Artifacts

**Problem**: MCQ performance may reflect ability to eliminate wrong answers rather than understand puns.

**Solution**:
- Use multiple evaluation formats and compare
- Analyze confidence scores, not just accuracy
- Check that performance is consistent across formats

---

## Advice for Interpretability Applications

When using model-written evaluations for interpretability research, consider these additional factors:

### Prioritize Token-Localized Formats

For most interpretability methods, cloze and MCQ formats are strictly superior to free-form generation:

| Method | Cloze/MCQ | Free-form |
|--------|-----------|-----------|
| Logit lens | Extract at decision token | Unclear where to extract |
| Activation patching | Patch at known position | Must search across positions |
| Probing | Clean single-position signal | Aggregation required |
| Attention analysis | Clear "what influences this token" | Diffuse across sequence |
| Causal tracing | Intervene at decision point | Multiple intervention points |

When generating evaluation data, convert free-form tasks to cloze format whenever possible:

- Instead of: "Explain why this is a pun"
- Use: "This is a pun because the word ___ has two meanings"

- Instead of: "What type of wordplay is this?"
- Use: "This pun uses: A) homophone B) polysemy C) syntactic ambiguity. Answer:"

### Choose Examples That Maximize Internal Signal

For probing and localization, you want examples where:
- The target concept is clearly present or clearly absent (avoid ambiguous cases)
- The contrast between positive and negative examples is minimal except for the concept of interest
- The model's behavior differs measurably between conditions

**For puns**: Pair each pun with a minimally different non-pun. "I lost interest" (pun about banking) vs. "I lost motivation" (not a pun). This controls for surface features and isolates the wordplay.

### Construct Minimal Pairs

A minimal pair differs in exactly one aspect. For cloze-style interpretability work, this is especially powerful:

```
Pun condition:
"I used to be a banker but I lost interest." 
This is a pun. Yes or No? Answer: [Yes]

Non-pun condition (minimal change):
"I used to be a banker but I lost patience."
This is a pun. Yes or No? Answer: [No]
```

Both prompts are nearly identical up to the decision token. Differences in activations at the final position can be attributed to the pun/non-pun distinction rather than confounding surface features.

When generating evaluation data, explicitly prompt for minimal pairs:

```
Generate pairs of sentences where one is a pun and one is not. 
The sentences should be as similar as possible, differing only in 
whether wordplay is present.

Pair 1:
Pun: "I used to be a banker but I lost interest."
Non-pun: "I used to be a banker but I lost patience."

Pair 2:
Pun: "The bicycle couldn't stand because it was two-tired."
Non-pun: "The bicycle couldn't stand because it was broken."

Generate 20 more minimal pairs:
```

### Generate Enough Examples for Statistical Power

Interpretability experiments often require hundreds or thousands of examples:
- Probing classifiers need training data
- Activation patching needs many trials to estimate causal effects
- PCA and other dimensionality reduction need sufficient samples

Model-written evaluation scales well here. Generate 500+ examples, filter to 200+ high-quality ones.

### Create Difficulty Gradations

Some puns are obvious; others are subtle. For interpretability, it helps to have:
- Easy examples (clear signal, useful for initial exploration)
- Hard examples (tests whether your methods capture genuine understanding)
- Edge cases (ambiguous cases that reveal the boundaries of model representations)

### Track Provenance

Record which examples came from which generation prompts. If you discover something interesting (e.g., the model fails on syntactic puns but succeeds on homophone puns), you can trace back to understand why.

---

## Example: A Complete Pun Evaluation Pipeline

Here is a concrete workflow for creating a pun evaluation dataset optimized for interpretability:

```
1. GENERATION PHASE
   ├── Generate 100 cloze-style pun recognition items (Yes/No format)
   ├── Generate 100 cloze-style explanation items (fill in the double meaning)
   ├── Generate 100 minimal pairs (pun + matched non-pun)
   ├── Generate 50 MCQ items testing mechanism understanding
   └── Generate 50 MCQ items testing wordplay identification

2. FILTERING PHASE
   ├── Remove duplicates (expect ~10-15% reduction)
   ├── Verify cloze answers are single tokens where possible
   ├── Check minimal pairs are actually minimal (edit distance)
   ├── Human review of 100 random samples
   │   └── Estimate quality rate, adjust generation if <80%
   └── Diversity analysis (topic distribution, mechanism types)

3. VALIDATION PHASE  
   ├── Test on held-out human annotators
   │   └── Do humans agree on which items are puns?
   ├── Pilot evaluation on 2-3 models of different sizes
   │   └── Check for ceiling/floor effects
   ├── Verify token localization
   │   └── Correct answers should be single tokens at predictable positions
   └── Iterate on generation prompts if needed

4. DEPLOYMENT PHASE
   ├── Split into train/validation/test (for probing)
   ├── For each example, record:
   │   ├── Full prompt text
   │   ├── Decision token position
   │   ├── Correct answer token(s)
   │   ├── Incorrect answer token(s) for comparison
   │   └── Minimal pair ID (if applicable)
   ├── Extract activations at decision positions during evaluation
   └── Run interpretability analyses (probing, patching, logit lens)
```

---

## Key Takeaways

1. **LLMs can generate evaluation data at scale**, dramatically reducing the bottleneck of dataset creation.

2. **Token localization is critical for interpretability**: design cloze and MCQ formats so the model's decision concentrates at a single, predictable position. This enables logit lens, activation patching, and probing at exactly the right location.

3. **Human oversight remains essential**: filter for quality, check for diversity, and validate that examples actually test what you intend.

4. **Construct minimal pairs**: examples that differ only in the presence or absence of the target concept provide the cleanest signal for comparing activations.

5. **Multiple evaluation formats** (zero-shot, ICL, MCQ) provide complementary signals and help identify format-specific artifacts.

6. **Watch for pitfalls**: memorization, surface shortcuts, and lack of diversity can all undermine your evaluation.

The Perez et al. methodology is a meta-technique: using AI capabilities to accelerate AI research. For interpretability researchers, it offers a path from "I want to study concept X" to "I have 500 high-quality examples of concept X" in hours rather than weeks.

---

## References

Perez, E., Ringer, S., Lukošiūtė, K., Nguyen, K., Chen, E., Heiner, S., Pettit, C., Olsson, C., Kundu, S., Kadavath, S., Jones, A., Chen, A., Mann, B., Israel, B., Seethor, B., McKinnon, C., Olah, C., Yan, D., Amodei, D., Amodei, D., Drain, D., Li, D., Tran-Schwartz, E., Khber, E., Showk, E., Lanham, T., Telleen-Lawton, T., Brown, T., Henighan, T., Hume, T., Bai, Y., Hatfield-Dodds, Z., Kaplan, J., Clark, J., Bowman, S. R., & Askell, A. (2022). Discovering Language Model Behaviors with Model-Written Evaluations. *arXiv preprint arXiv:2212.09251*.
