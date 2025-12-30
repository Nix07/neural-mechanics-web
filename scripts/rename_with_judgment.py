#!/usr/bin/env python3
"""
Rename arxiv papers with human-judged short titles.
Format: firstauthorlastname-year-shorttitle-arxivid.pdf
"""

import os
from pathlib import Path

PAPERS_DIR = Path(__file__).parent.parent / "papers"

# Map current names to better short titles (based on actual paper content/common names)
RENAMES = {
    # 2013
    "mikolov-2013-efficient-estimation-word-representations-1301.3781.pdf":
        "mikolov-2013-word2vec-1301.3781.pdf",

    # 2015
    "andreas-2015-neural-module-networks-1511.02799.pdf":
        "andreas-2015-neural-module-networks-1511.02799.pdf",  # keep

    # 2016
    "ribeiro-2016-why-should-i-trust-1602.04938.pdf":
        "ribeiro-2016-lime-1602.04938.pdf",
    "lipton-2016-mythos-model-interpretability-1606.03490.pdf":
        "lipton-2016-mythos-interpretability-1606.03490.pdf",
    "bolukbasi-2016-man-computer-programmer-woman-1607.06520.pdf":
        "bolukbasi-2016-debiasing-embeddings-1607.06520.pdf",

    # 2017
    "doshi-velez-2017-towards-rigorous-science-interpretable-1702.08608.pdf":
        "doshi-velez-2017-rigorous-interpretability-1702.08608.pdf",
    "sundararajan-2017-axiomatic-attribution-deep-networks-1703.01365.pdf":
        "sundararajan-2017-integrated-gradients-1703.01365.pdf",
    "lundberg-2017-unified-approach-interpreting-model-1705.07874.pdf":
        "lundberg-2017-shap-1705.07874.pdf",
    "kim-2017-interpretability-beyond-feature-attribution-1711.11279.pdf":
        "kim-2017-tcav-1711.11279.pdf",

    # 2018 - Note: 1801.10242 seems to be wrong paper
    "mueller-2018-lowrank-bandit-methods-highdimensional-1801.10242.pdf":
        None,  # DELETE - wrong paper
    "hooker-2018-benchmark-interpretability-methods-deep-1806.10758.pdf":
        "hooker-2018-roar-benchmark-1806.10758.pdf",
    "adebayo-2018-sanity-checks-saliency-maps-1810.03292.pdf":
        "adebayo-2018-sanity-checks-1810.03292.pdf",
    "rudin-2018-stop-explaining-black-box-1811.10154.pdf":
        "rudin-2018-stop-explaining-blackbox-1811.10154.pdf",
    "bau-2018-gan-dissection-visualizing-understanding-1811.10597.pdf":
        "bau-2018-gan-dissection-1811.10597.pdf",

    # 2019
    "jain-2019-attention-not-explanation-1902.10186.pdf":
        "jain-2019-attention-not-explanation-1902.10186.pdf",  # keep
    "tenney-2019-bert-rediscovers-classical-nlp-1905.05950.pdf":
        "tenney-2019-bert-pipeline-1905.05950.pdf",
    "tenney-2019-what-do-you-learn-1905.06316.pdf":
        "tenney-2019-edge-probing-1905.06316.pdf",
    "wiegreffe-2019-attention-not-not-explanation-1908.04626.pdf":
        "wiegreffe-2019-attention-not-not-explanation-1908.04626.pdf",  # keep
    "petroni-2019-language-models-knowledge-bases-1909.01066.pdf":
        "petroni-2019-lama-1909.01066.pdf",
    "hewitt-2019-designing-interpreting-probes-control-1909.03368.pdf":
        "hewitt-2019-control-tasks-1909.03368.pdf",
    "slack-2019-fooling-lime-shap-adversarial-1911.02508.pdf":
        "slack-2019-fooling-lime-shap-1911.02508.pdf",

    # 2020
    "nozza-2020-what-mask-making-sense-2003.02912.pdf":
        None,  # DELETE - wrong paper, not relevant
    "samek-2020-explaining-deep-neural-networks-2003.07631.pdf":
        "samek-2020-xai-review-2003.07631.pdf",
    "pimentel-2020-informationtheoretic-probing-linguistic-structure-2004.03061.pdf":
        "pimentel-2020-info-theoretic-probing-2004.03061.pdf",
    "jacovi-2020-towards-faithfully-interpretable-nlp-2004.03685.pdf":
        "jacovi-2020-faithfulness-2004.03685.pdf",
    "abnar-2020-quantifying-attention-flow-transformers-2005.00928.pdf":
        "abnar-2020-attention-flow-2005.00928.pdf",
    "brown-2020-language-models-fewshot-learners-2005.14165.pdf":
        "brown-2020-gpt3-2005.14165.pdf",
    "geva-2020-transformer-feedforward-layers-keyvalue-2012.14913.pdf":
        "geva-2020-ffn-key-value-2012.14913.pdf",

    # 2021
    "bolukbasi-2021-interpretability-illusion-bert-2104.07143.pdf":
        "bolukbasi-2021-interpretability-illusion-2104.07143.pdf",
    "dodge-2021-documenting-large-webtext-corpora-2104.08758.pdf":
        None,  # DELETE - not interpretability
    "mou-2021-narrative-question-answering-cuttingedge-2106.03826.pdf":
        None,  # DELETE - wrong paper
    "hu-2021-lora-lowrank-adaptation-large-2106.09685.pdf":
        "hu-2021-lora-2106.09685.pdf",
    "lin-2021-truthfulqa-measuring-how-models-2109.07958.pdf":
        "lin-2021-truthfulqa-2109.07958.pdf",

    # 2022
    "power-2022-grokking-generalization-beyond-overfitting-2201.02177.pdf":
        "power-2022-grokking-2201.02177.pdf",
    "jia-2022-electricmagnetic-duality-mathbbz2-symmetry-2201.12361.pdf":
        None,  # DELETE - wrong paper (physics)
    "meng-2022-locating-editing-factual-associations-2202.05262.pdf":
        "meng-2022-rome-2202.05262.pdf",
    "wei-2022-emergent-abilities-large-language-2206.07682.pdf":
        "wei-2022-emergent-abilities-2206.07682.pdf",
    "elhage-2022-toy-models-superposition-2209.10652.pdf":
        "elhage-2022-superposition-2209.10652.pdf",
    "olsson-2022-incontext-learning-induction-heads-2209.11895.pdf":
        "olsson-2022-induction-heads-2209.11895.pdf",
    "meng-2022-massediting-memory-transformer-2210.07229.pdf":
        "meng-2022-memit-2210.07229.pdf",
    "xu-2022-autonomous-quantum-error-correction-2210.13406.pdf":
        None,  # DELETE - wrong paper (quantum)
    "wang-2022-interpretability-wild-circuit-indirect-2211.00593.pdf":
        "wang-2022-ioi-circuit-2211.00593.pdf",
    "liang-2022-holistic-evaluation-language-models-2211.09110.pdf":
        "liang-2022-helm-2211.09110.pdf",
    "perez-2022-discovering-language-model-behaviors-2212.09251.pdf":
        "perez-2022-model-written-evals-2212.09251.pdf",

    # 2023
    "nanda-2023-progress-measures-grokking-via-2301.05217.pdf":
        "nanda-2023-grokking-progress-2301.05217.pdf",
    "gurrapu-2023-rationalization-explainable-nlp-survey-2301.08912.pdf":
        "gurrapu-2023-rationalization-survey-2301.08912.pdf",
    "sarti-2023-inseq-interpretability-toolkit-sequence-2302.13942.pdf":
        "sarti-2023-inseq-2302.13942.pdf",
    "shin-2023-superhuman-artificial-intelligence-can-2303.07462.pdf":
        "shin-2023-superhuman-ai-decisions-2303.07462.pdf",
    "belrose-2023-eliciting-latent-predictions-transformers-2303.08112.pdf":
        "belrose-2023-tuned-lens-2303.08112.pdf",
    "goldowsky-dill-2023-localizing-model-behavior-path-2304.05969.pdf":
        "goldowsky-dill-2023-path-patching-2304.05969.pdf",
    "geva-2023-dissecting-recall-factual-associations-2304.14767.pdf":
        "geva-2023-dissecting-factual-recall-2304.14767.pdf",
    "conmy-2023-towards-automated-circuit-discovery-2304.14997.pdf":
        "conmy-2023-acdc-2304.14997.pdf",
    "wu-2023-interpretability-scale-identifying-causal-2305.08809.pdf":
        "wu-2023-boundless-das-2305.08809.pdf",
    "fu-2023-improving-language-model-negotiation-2305.10142.pdf":
        None,  # DELETE - wrong paper
    "wang-2023-gaussian-process-probes-gpp-2305.18213.pdf":
        None,  # DELETE - not core interpretability
    "li-2023-inferencetime-intervention-eliciting-truthful-2306.03341.pdf":
        "li-2023-iti-2306.03341.pdf",
    "zhu-2023-promptrobust-towards-evaluating-robustness-2306.04528.pdf":
        None,  # DELETE - wrong paper
    "zheng-2023-judging-llmasajudge-mtbench-chatbot-2306.05685.pdf":
        "zheng-2023-llm-as-judge-2306.05685.pdf",
    "hernandez-2023-linearity-relation-decoding-transformer-2308.09124.pdf":
        "hernandez-2023-linear-relations-2308.09124.pdf",
    "turner-2023-steering-language-models-activation-2308.10248.pdf":
        "turner-2023-activation-steering-2308.10248.pdf",
    "cunningham-2023-sparse-autoencoders-find-highly-2309.08600.pdf":
        "cunningham-2023-sparse-autoencoders-2309.08600.pdf",
    "sarti-2023-quantifying-plausibility-context-reliance-2310.01188.pdf":
        "sarti-2023-pecore-2310.01188.pdf",
    "zou-2023-representation-engineering-topdown-approach-2310.01405.pdf":
        "zou-2023-representation-engineering-2310.01405.pdf",
    "mcdougall-2023-copy-suppression-comprehensively-understanding-2310.04625.pdf":
        "mcdougall-2023-copy-suppression-2310.04625.pdf",
    "marks-2023-geometry-truth-emergent-linear-2310.06824.pdf":
        "marks-2023-geometry-of-truth-2310.06824.pdf",
    "sachdeva-2023-farzi-data-autoregressive-data-2310.09983.pdf":
        None,  # DELETE - wrong paper
    "tigges-2023-linear-representations-sentiment-large-2310.15154.pdf":
        "tigges-2023-sentiment-directions-2310.15154.pdf",
    "todd-2023-function-vectors-large-language-2310.15213.pdf":
        "todd-2023-function-vectors-2310.15213.pdf",
    "schut-2023-bridging-humanai-knowledge-gap-2310.16410.pdf":
        "schut-2023-alphazero-concepts-2310.16410.pdf",
    "zhou-2023-dont-make-your-llm-2311.01964.pdf":
        None,  # DELETE - not core interpretability

    # 2024
    "ghandeharioun-2024-patchscopes-unifying-framework-inspecting-2401.06102.pdf":
        "ghandeharioun-2024-patchscopes-2401.06102.pdf",
    "wendler-2024-do-llamas-work-english-2402.10588.pdf":
        "wendler-2024-latent-language-2402.10588.pdf",
    "prakash-2024-finetuning-enhances-existing-mechanisms-2402.14811.pdf":
        "prakash-2024-entity-tracking-2402.14811.pdf",
    "xu-2024-multidisciplinary-framework-deconstructing-bots-2402.15119.pdf":
        None,  # DELETE - wrong paper
    "hanna-2024-have-faith-faithfulness-going-2403.17806.pdf":
        "hanna-2024-eap-ig-2403.17806.pdf",
    "ferrando-2024-primer-inner-workings-transformerbased-2405.00208.pdf":
        "ferrando-2024-transformer-primer-2405.00208.pdf",
    "gao-2024-scaling-evaluating-sparse-autoencoders-2406.04093.pdf":
        "gao-2024-scaling-saes-2406.04093.pdf",
    "gandelsman-2024-interpreting-secondorder-effects-neurons-2406.04341.pdf":
        "gandelsman-2024-clip-neurons-2406.04341.pdf",
    "dunefsky-2024-transcoders-find-interpretable-llm-2406.11944.pdf":
        "dunefsky-2024-transcoders-2406.11944.pdf",
    "qi-2024-model-internalsbased-answer-attribution-2406.13663.pdf":
        "qi-2024-mirage-2406.13663.pdf",
    "ren-2024-first-running-time-analysis-2406.16116.pdf":
        None,  # DELETE - wrong paper (evolutionary algorithms)
    "paulo-2024-automatically-interpreting-millions-features-2410.13928.pdf":
        "paulo-2024-auto-interp-2410.13928.pdf",
    "roy-2024-uniform-discretized-integrated-gradients-2412.03886.pdf":
        "roy-2024-udig-2412.03886.pdf",

    # 2025
    "sharkey-2025-open-problems-mechanistic-interpretability-2501.16496.pdf":
        "sharkey-2025-open-problems-2501.16496.pdf",
    "hewitt-2025-we-cant-understand-ai-2502.07586.pdf":
        "hewitt-2025-new-vocabulary-2502.07586.pdf",
    "li-2025-interpretability-illusions-sparse-autoencoders-2505.16004.pdf":
        "li-2025-sae-illusions-2505.16004.pdf",
    "kim-2025-because-we-have-llms-2506.12152.pdf":
        "kim-2025-agentic-interpretability-2506.12152.pdf",
    "hewitt-2025-neologism-learning-controllability-selfverbalization-2510.08506.pdf":
        "hewitt-2025-neologism-learning-2510.08506.pdf",
}

def main():
    to_delete = []
    to_rename = []

    for old_name, new_name in RENAMES.items():
        old_path = PAPERS_DIR / old_name
        if not old_path.exists():
            print(f"SKIP (not found): {old_name}")
            continue

        if new_name is None:
            to_delete.append(old_name)
            print(f"DELETE: {old_name}")
        elif old_name != new_name:
            to_rename.append((old_name, new_name))
            print(f"RENAME: {old_name}")
            print(f"    ->  {new_name}")

    print(f"\n{len(to_delete)} files to delete, {len(to_rename)} files to rename")

    # Execute deletions
    for filename in to_delete:
        path = PAPERS_DIR / filename
        os.remove(path)
        print(f"Deleted: {filename}")

    # Execute renames
    for old_name, new_name in to_rename:
        old_path = PAPERS_DIR / old_name
        new_path = PAPERS_DIR / new_name
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")

    print("\nDone!")

if __name__ == "__main__":
    main()
