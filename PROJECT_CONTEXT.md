# AI6130 Course Project — Project Context

> **Purpose:** This is the durable handover document for the Codex project. Read it before planning experiments, editing the proposal, or writing code. It distinguishes confirmed decisions from working choices that still require agreement.

## 1. Project identity

- **Course:** AI6130 Large Language Models
- **Team:** G4
- **Instructor:** Prof. Luu Anh Tuan
- **Final project title:** **Can Vulnerability-Guided KL Anchors Reduce Catastrophic Forgetting in Parameter-Efficient Fine-Tuning?**

### Team members

| Name | Matriculation number |
|---|---|
| Sridhar Srihari | G2510717L |
| Dadi Hemanth | G2510794D |
| Siddharth Paliwal | G2510801L |
| Balasubramanian Hariharan | G2610112K |

## 2. One-paragraph project summary

Fine-tuning a language model for a narrow target task can improve that task while reducing performance on unrelated capabilities, a phenomenon commonly called catastrophic forgetting. This project investigates whether a limited KL-regularization budget can be allocated more effectively by selecting general-purpose prompts that appear behaviourally vulnerable during an initial task-adaptation stage. The initial target is mathematical adaptation with GSM8K, while retention will be measured on BoolQ, HellaSwag and ARC-Easy. Vulnerability-guided KL anchoring will be compared with an untouched base model, task-only LoRA, random KL-anchor selection and semantic-diversity-based KL-anchor selection. The central question is whether vulnerability-guided selection preserves more non-target benchmark performance at comparable GSM8K performance and under the same anchor-token budget.

## 3. Plain-language explanation

Imagine that the language model is a student who already knows many subjects. We want it to become better at mathematics, but intensive mathematics practice may make some of its other behaviours worse.

We retain an untouched copy of the original model as a **teacher**. Another copy receives a LoRA adapter and learns GSM8K mathematics. A preliminary adaptation can be used as a diagnostic: we show the original teacher and the task-adapted model many general-purpose prompts and identify prompts on which their token predictions have changed the most. These prompts act like warning lights indicating behaviours that may be vulnerable to adaptation.

The final comparison models must start independently from the same untouched checkpoint. During vulnerability-guided training, the model learns GSM8K while also seeing selected anchor prompts. On those prompts, a KL-divergence loss encourages its token-probability distribution to remain close to the frozen teacher.

We compare this strategy with choosing the same amount of anchor material randomly and choosing semantically diverse anchors. This isolates the main research question: **does information about observed vulnerability help us spend a limited preservation budget more effectively?**

## 4. Important terminology

- **Base model / reference model / teacher:** The untouched instruction-tuned checkpoint. It remains frozen and supplies reference token probabilities.
- **Student:** A copy of the same checkpoint with trainable LoRA adapters.
- **LoRA:** Parameter-efficient fine-tuning in which the original backbone remains frozen and small low-rank adapter matrices are trained.
- **Anchor prompt:** A general-purpose prompt used during adaptation to discourage excessive drift from the reference model.
- **Unlabelled anchor:** An anchor does not require a human-written correct answer. The teacher's soft token-probability distribution supplies the preservation target.
- **Vulnerability or damage score:** The measured divergence between the frozen teacher and a task-adapted model on a candidate prompt.
- **Semantic diversity:** Diversity in meaning rather than surface wording. It can be approximated by embedding prompts, clustering similar prompts and selecting representatives from different clusters.
- **Catastrophic forgetting in this project:** A reduction from the base checkpoint's performance to the adapted checkpoint's performance on the selected non-target benchmarks.

## 5. Primary research question

> Under a fixed anchor-token budget and at approximately comparable GSM8K performance, do general-purpose prompts selected using adaptation-induced behavioural divergence preserve BoolQ, HellaSwag and ARC-Easy performance better than random or semantic-diversity-based KL-anchor selection during LoRA fine-tuning?

### Primary hypothesis

Vulnerability-guided anchors will produce a better target-learning–retention trade-off than task-only LoRA and equally budgeted random or diversity-based anchoring.

### Secondary questions

1. Does the vulnerability score measured during the preliminary adaptation predict later non-target performance loss?
2. Are any retention benefits consistent across BoolQ, HellaSwag and ARC-Easy, or limited to particular behaviours?
3. How much target-task performance is sacrificed, if any, to obtain better retention?
4. Are conclusions stable across random seeds?

A KL-weight sweep, multiple preliminary seeds or additional adaptation domains are useful stretch goals, not required core experiments.

## 6. What the proposed contribution is—and is not

The project does **not** claim to invent:

- LoRA;
- knowledge distillation;
- KL regularization;
- learning without forgetting; or
- the general observation that LoRA can forget less than full fine-tuning.

The proposed course-level contribution is:

> A controlled investigation of whether behavioural drift observed during task adaptation is a useful signal for selecting a limited set of KL anchors, compared with random and semantic-diversity selection under equal budgets.

All anchored conditions should use the same KL mechanism. Only the anchor-selection strategy should change. This is essential because it isolates the value of the vulnerability signal.

Any novelty claim must remain cautious until the final literature-gap check verifies that an equivalent selection protocol has not already been evaluated. A negative result remains scientifically useful: it would show that high preliminary drift is not necessarily a good guide for allocating preservation regularization.

## 7. Confirmed experimental scope

### Initial target task

- **Dataset:** GSM8K
- **Purpose:** LoRA fine-tuning for mathematical word-problem solving
- **Primary target metric:** Exact-match accuracy of the final numerical answer

The project may be extended to another adaptation domain only if time and compute permit. Such an extension is not part of the minimum viable project.

### Non-target retention benchmarks

- **BoolQ:** Binary reading-comprehension/question-answering accuracy
- **HellaSwag:** Commonsense completion, evaluated using a locked multiple-choice likelihood protocol
- **ARC-Easy:** Elementary-science multiple-choice accuracy

These benchmarks measure selected non-target behaviours; they do not represent every general capability of an LLM. Formal claims should therefore refer to **retention on the selected non-target benchmarks**, not preservation of all general knowledge or generalization.

### Model family

`Qwen/Qwen2.5-1.5B-Instruct` is the selected model repository. Its immutable revision remains unresolved; see Section 15 and `DECISIONS.md`.

## 8. Core experimental conditions

| Condition | GSM8K training | KL anchors | Purpose |
|---|---:|---:|---|
| Untouched base model | No | No | Reference scores before adaptation |
| Task-only LoRA | Yes | No | Main adaptation baseline |
| Random-anchor LoRA | Yes | Random selection | Controls for receiving KL regularization |
| Diversity-anchor LoRA | Yes | Semantic-diversity selection | Tests broad prompt coverage |
| Vulnerability-anchor LoRA | Yes | Highest measured drift | Proposed method |

The combined **diversity + vulnerability** strategy is implemented as an optional extension. The original core comparisons remain the required study.

## 9. Working technical methodology

The high-level study is agreed, while implementation details marked as open in `DECISIONS.md` still require team confirmation.

### Stage A — Establish the reference

1. Load the untouched instruction-model checkpoint.
2. Fix the tokenizer, chat template, decoding rules, evaluation-harness version and prompt templates.
3. Evaluate the base checkpoint on GSM8K, BoolQ, HellaSwag and ARC-Easy.
4. Store item-level predictions, likelihood scores where applicable, parse failures and aggregate metrics.
5. Do not use a retention task as a primary forgetting benchmark if the base model performs at or near chance, because there would be little demonstrated capability to forget.

### Stage B — Build a disjoint candidate-anchor pool

1. Collect general instruction prompts from a permitted public dataset. Dolly 15k has been discussed as one possible source, but the source is not yet final.
2. Remove empty, unsafe, excessively long, duplicate and strongly GSM8K-like prompts.
3. Remove exact and high-similarity overlap with GSM8K validation/test items and with BoolQ, HellaSwag and ARC-Easy evaluation items.
4. Freeze the cleaned candidate pool before examining final results.
5. Keep final benchmark examples completely separate from anchor selection and training.

A working pool size of roughly 2,000 prompts has been discussed but is not yet a final decision.

### Stage C — Produce behavioural vulnerability scores

The method requires a task-adapted checkpoint against which teacher–student drift can be measured. The exact duration and construction of that discovery stage are still open.

1. Starting from the reference checkpoint, train a temporary LoRA adapter on a fixed GSM8K subset for a pre-specified amount of adaptation.
2. For every candidate prompt, generate and save a short deterministic teacher continuation, or otherwise define one fixed continuation sequence.
3. Force the frozen teacher and temporary adapted model to score the **same continuation tokens**.
4. At every scored response position, compare their full next-token probability distributions.
5. Average teacher-to-adapted-model KL divergence across valid response tokens:

\[
D(x)=\frac{1}{T_x}\sum_{t=1}^{T_x}
D_{\mathrm{KL}}\!\left(
p_{\mathrm{teacher}}(\cdot\mid x,y_{<t})
\parallel
p_{\mathrm{adapted}}(\cdot\mid x,y_{<t})
\right)
\]

6. Treat a high score as evidence of substantial behavioural drift, not proof that the model became factually incorrect.
7. Do not use this temporary adapter to initialize any final comparison model. Every final condition must restart independently from the same untouched checkpoint.

### Stage D — Select equally budgeted anchor sets

- **Random condition:** Select prompts uniformly from the cleaned candidate pool.
- **Diversity condition:** Embed the prompts, group semantically similar prompts and select representatives across groups.
- **Vulnerability condition:** Select prompts with the highest damage scores.
- Allocate the same total number of scored anchor tokens to every anchored condition. Equal prompt counts are insufficient when prompt and response lengths differ.
- Reuse the same saved teacher targets and scoring procedure across selection methods.

### Stage E — Conduct fresh final LoRA runs

For each final condition:

1. Restart from the identical untouched reference checkpoint.
2. Use the same GSM8K training examples, data-order or controlled-seed policy, LoRA configuration, optimizer, learning-rate schedule, training duration and prompt template.
3. For anchored conditions, interleave GSM8K learning with anchor KL regularization.
4. Train only the student's LoRA parameters; keep the backbone frozen.
5. Keep the teacher completely frozen.

The anchored training objective is:

\[
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{GSM8K}}
+
\lambda\mathcal{L}_{\text{anchor-KL}}
\]

The first term teaches mathematics. The second discourages departure from the teacher on selected anchors.

### Stage F — Evaluate target learning and retention

Run the same locked evaluation protocol on every final checkpoint. Preserve raw item-level outputs so that aggregate results can be audited and qualitative errors can be examined.

## 10. What KL anchoring actually does

KL anchoring does not compare only two generated sentences. It compares the teacher's and student's next-token probability distributions under the same prompt and response prefix.

An anchor-generated gradient is not stored only for the prompt that produced it. The gradient updates the same shared LoRA matrices used for all inputs. Consequently, it can affect many related and unrelated prompts. The hope is that carefully selected anchors produce useful preservation constraints that generalize beyond the selected examples.

The project is **not** identifying the exact parameters that contain mathematics or the exact parameters damaged by mathematics. It measures vulnerability at the behavioural prompt level. True parameter localization would require a different mechanistic project using gradients, activations, attribution or causal intervention.

## 11. Evaluation and reporting

### Target learning

\[
\Delta_{\text{target}}
=
S_{\text{GSM8K, adapted}}
-
S_{\text{GSM8K, base}}
\]

Primary target metric: GSM8K final-answer exact match.

### Forgetting

For each non-target benchmark \(b\):

\[
F_b
=
S_{b,\text{base}}
-
S_{b,\text{adapted}}
\]

- \(F_b>0\): performance decreased.
- \(F_b=0\): no measured change.
- \(F_b<0\): performance improved after adaptation.

Report individual benchmark changes rather than hiding them behind only one average.

### Benchmark metrics

- **GSM8K:** Final numerical-answer exact match
- **BoolQ:** Accuracy
- **HellaSwag:** Standard multiple-choice accuracy; report length-normalized scoring when supplied by the locked evaluation harness
- **ARC-Easy:** Standard multiple-choice accuracy; report the pre-specified normalized variant when applicable

The exact evaluation-harness tasks and whether `acc` or `acc_norm` is primary must be fixed before final experiments.

### Main visualizations

1. GSM8K performance versus average non-target forgetting
2. Per-benchmark forgetting for every method
3. Target-learning–retention Pareto plot
4. Vulnerability score versus later behavioural change, if a defensible item- or group-level correspondence can be constructed

### Qualitative analysis

Inspect examples where:

- task-only LoRA fails but an anchored model retains the base behaviour;
- vulnerability anchoring differs from random or diversity anchoring;
- apparent forgetting is caused by formatting or response-style changes; and
- the teacher itself is uncertain or incorrect.

## 12. Fairness and experimental controls

The following controls are essential:

1. **Same starting checkpoint:** Every final run starts from the same untouched model.
2. **Same target data:** Use identical GSM8K training data and splits.
3. **Same LoRA and optimization configuration:** Anchor selection should be the primary manipulated variable.
4. **Equal anchor-token budget:** Random, diversity and vulnerability conditions receive the same preservation-token allocation.
5. **Comparable compute among anchored methods:** Record training time and extra teacher-forward cost.
6. **Target-performance matching:** Compare forgetting at approximately equal GSM8K validation performance. A model that forgets less merely because it learned less mathematics is not automatically better.
7. **Trade-off reporting:** Report GSM8K improvement and forgetting together rather than presenting retention alone.
8. **No benchmark leakage:** Retention test questions must never become anchors.
9. **Locked protocol:** Decide prompts, scoring, metrics, seeds and checkpoint-selection rules before examining final test outcomes.
10. **Multiple seeds if feasible:** Prefer multiple final seeds; otherwise label results as preliminary and avoid strong significance claims.

## 13. Interpretation risks

- High KL divergence can reflect harmless formatting or stylistic drift rather than lost knowledge.
- The teacher can be wrong. KL anchoring may preserve its errors as well as its useful behaviour.
- A high-drift prompt may be unstable because the teacher is uncertain, not because it represents an important capability.
- Retention can improve simply because the KL term prevents the model from learning GSM8K strongly; target-performance matching is therefore important.
- Results on three benchmarks do not justify claims about all general capabilities.
- A selection method tuned on the final test set would invalidate the comparison.
- Prompt-level vulnerability is not parameter-level localization.
- Negative forgetting values should be reported as improvement rather than clipped without explanation.
- If base performance is close to chance, score differences may be noisy and difficult to interpret.

## 14. Decisions that are final

- The project is research-oriented rather than a simple application.
- The title and central topic are finalized.
- The core intervention is vulnerability-guided KL-anchor selection during LoRA fine-tuning.
- Mathematics with GSM8K is the initial adaptation domain.
- BoolQ, HellaSwag and ARC-Easy are the planned non-target retention benchmarks.
- Core comparisons are the base model, task-only LoRA, random KL anchors, diversity-based KL anchors and vulnerability-guided KL anchors.
- The core study will use a Qwen instruction model in the 1.5B range.
- Other target domains are optional extensions only.
- The combined diversity-plus-damage selector is not part of the core project.
- Formal claims must concern retention on the selected benchmarks, not preservation of every general capability.

## 15. Choices still unresolved

These must be recorded in an experiment protocol before final training:

1. **Exact checkpoint revision**
   - The repository ID is `Qwen/Qwen2.5-1.5B-Instruct` by user decision on 2026-09-16. Pin its immutable commit revision before reported runs.

2. **Anchor-pool source and licence**
3. **Candidate-pool size**
4. **Length and dataset fraction of the discovery adaptation**
5. **LoRA rank, alpha, dropout, target modules and precision**
6. **Whether to use standard LoRA or QLoRA for memory feasibility**
7. **Number of anchor tokens and anchor-interleaving frequency**
8. **KL direction, temperature, normalization and weight \(\lambda\)**
9. **Teacher-continuation length and decoding settings**
10. **Evaluation harness/version, chat template, few-shot setting and `acc` versus `acc_norm` choice**
11. **Number of random seeds**
12. **Rule for matching GSM8K performance across conditions**
13. **Whether an aggregate forgetting score will be reported and how it will be normalized**
14. **Available GPU model, memory, storage and total compute budget**

## 16. Earlier ideas that are not the finalized project

The following were explored during discussion but should not be confused with the selected study:

- A direct “Does LoRA forget less than full fine-tuning?” replication
- A mechanistic study separating low-rank constraints, frozen backbones and update magnitude
- Full fine-tuning versus LoRA with and without generic knowledge-distillation regularization
- A primary comparison of O-LoRA, OPLoRA and CURLoRA
- Discovering and protecting the exact parameters damaged by mathematics
- Using ARC-Challenge instead of the finalized ARC-Easy core benchmark
- Making combined vulnerability-and-diversity selection a required condition
- Treating additional target domains as mandatory

These ideas may inform discussion or future work but should not expand the minimum viable experiment.

## 17. Communication note about the vulnerability-discovery stage

The concise proposal and professor email intentionally omit the term **“pilot.”** Internally, however, the vulnerability selector still needs a preliminary or discovery adaptation to create a task-adapted model against which teacher–student drift can be measured.

That discovery model should be:

- temporary;
- used only for vulnerability scoring;
- excluded as an initialization for all final comparison runs; and
- fully specified before implementation.

If the team removes the preliminary run from the actual method, it must define another source of vulnerability scores. Removing only the word does not remove the technical requirement.

## 18. Immediate next steps

1. Pin the exact `Qwen/Qwen2.5-1.5B-Instruct` revision.
2. Conduct a focused literature-gap check for behaviour-drift-guided replay or KL-anchor selection.
3. Write and freeze an experimental protocol covering every unresolved choice in Section 15.
4. Confirm hardware feasibility with a small forward/backward-pass smoke test.
5. Reproduce base-model evaluation on all four benchmarks.
6. Implement and validate task-only GSM8K LoRA as the main baseline.
7. Build and decontaminate the candidate-anchor pool.
8. Cache teacher continuations/logits or determine a storage-efficient equivalent.
9. Produce vulnerability scores using the agreed discovery protocol.
10. Construct equal-token random, diversity and vulnerability anchor sets.
11. Run the final controlled experiments.
12. Produce target-retention plots, seed-level results and qualitative error analysis.
13. Document both positive and negative findings without changing metrics after seeing test results.

## 19. Suggested minimum viable project

If compute or time becomes restrictive, retain:

- one exact 1.5B Qwen checkpoint;
- GSM8K as the target task;
- BoolQ, HellaSwag and ARC-Easy as retention benchmarks;
- base, task-only, random-anchor, diversity-anchor and vulnerability-anchor conditions;
- one pre-specified KL weight;
- equal anchor-token budgets;
- at least one complete seed, preferably more;
- target-performance-aware comparison; and
- a transparent limitations section.

Do not sacrifice the task-only baseline, random baseline, fixed budget or contamination controls. These are more important to the research claim than adding extra models or datasets.

## 20. Key references from the proposal discussion

1. Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models.** ICLR, 2022.
2. Biderman et al. **LoRA Learns Less and Forgets Less.** TMLR, 2024.
3. Hinton, Vinyals and Dean. **Distilling the Knowledge in a Neural Network.** 2015.
4. Li and Hoiem. **Learning without Forgetting.** ECCV, 2016.
5. Jin and Ren. **Demystifying Language Model Forgetting with Low-rank Example Associations.** NeurIPS, 2025.
6. Zheng et al. **Spurious Forgetting in Continual Learning of Language Models.** ICLR, 2025.

Bibliographic details and links should be checked against the literature-review files before final report submission.
