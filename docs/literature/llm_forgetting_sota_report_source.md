# Catastrophic Forgetting and Capability Degradation in LLM Fine-Tuning

## State of the art, research gaps, and eight-week project proposals

**Coverage date:** 4 September 2026  
**Audience:** 3-4 MSAI students selecting an eight-week LLM course project  
**Primary focus:** decoder-only language models, LoRA and other PEFT methods, full-parameter fine-tuning, and continual instruction tuning  

## Decision first

The original idea, **"Does LoRA forget less than full fine-tuning?"**, should not be presented as novel research. It is a good replication question, but the main observation was studied directly by Biderman et al. in [LoRA Learns Less and Forgets Less](https://openreview.net/forum?id=aloEru2qCG), and later work examined update rank, spectral intruders, frozen backbones, layer location, orthogonality, magnitude, and several LoRA variants. A plain "LoRA plus knowledge distillation" proposal is also no longer novel because KL anchoring, self-distillation, real and synthetic replay, and small old-prompt sets have all been studied.

The best course project is:

### Replay or Re-alignment? Damage-Aware KL Anchoring for Small Language Models

**Research question.** Under a fixed anchor-token and training-compute budget, can a cheap pilot run identify general prompts that are especially vulnerable to fine-tuning, so that KL anchoring on those prompts preserves more capability than random anchoring? Does it reduce durable capability loss, or only prompt and task-alignment failure?

**Core idea.** Start with a separate pool of unlabeled, general prompts. After a short target-task warm-up, measure how much each prompt's output distribution has drifted from the base model. Select either the most damaged prompts or a diverse subset of them. During the main LoRA run, penalize divergence from the frozen base model on that small anchor set. Compare this with task-only LoRA, random KL anchors, random labeled replay, and a diversity-only anchor baseline. Evaluate ordinary benchmark loss and recovery under equivalent prompt templates, forced-choice likelihood, few-shot demonstrations, and a tiny alignment-only repair set.

**Honest novelty claim.** Replay selection, KL anchoring, synthetic replay, example-level forgetting prediction, and spurious forgetting all exist separately. This review did not find a controlled decoder-only study that combines: (1) pilot-damage selection of unlabeled KL anchors, (2) a fixed anchor-token budget, (3) matched target-task improvement, and (4) a recovery-aware decomposition of observed forgetting. Novelty confidence is **medium**, not absolute. A final pre-registration search should be run immediately before implementation.

**Why it fits the course.** It needs no new model architecture, works with a 0.5B-1.5B model, produces interpretable negative results, and fits 12-18 central runs. The minimum viable version is a simpler recovery-aware comparison of task-only LoRA, random KL anchoring, and damage-aware KL anchoring on one model and one target task.

## Executive summary

### Findings that are reasonably established

1. **Fine-tuning can reduce capabilities outside the target distribution.** This occurs in single-stage specialization, sequential instruction tuning, continual pretraining, safety tuning, and multimodal adaptation. The size and meaning of the drop depend strongly on task, prompt, checkpoint, and metric. Evidence includes the controlled and broad studies by [Luo et al.](https://arxiv.org/abs/2308.08747), [TRACE](https://arxiv.org/abs/2310.06762), and [Mapping Post-Training Forgetting at Scale](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bc1c5e5fb8ed1ef9b9b5abced2022e40-Abstract-Conference.html).

2. **LoRA is not immune to forgetting.** Low rank often reduces both target learning and forgetting, but this is a trade-off, not a guarantee. Rank, alpha, learning rate, targeted modules, training duration, and update norm can reverse apparent conclusions. [Biderman et al.](https://openreview.net/forum?id=aloEru2qCG) and [Shuttleworth et al.](https://papers.nips.cc/paper_files/paper/2025/hash/ff541950d1e885af90f523571564a401-Abstract-Conference.html) provide the strongest direct evidence.

3. **A frozen backbone does not guarantee preserved behavior while an adapter is active.** The original weights remain recoverable by disabling the adapter, but the adapted forward pass can still distort activations and outputs. Recent PEFT work ties retention to update geometry, activation drift, and singular directions, not merely the number of frozen parameters.

4. **Raw benchmark drops mix several phenomena.** Some observed forgetting is prompt-format or task-alignment failure rather than complete knowledge erasure. Equivalent prompts, forced-choice scoring, function-vector interventions, or small recovery sets can restore performance in some settings. This is supported by [Implicit Inference](https://proceedings.iclr.cc/paper_files/paper/2024/hash/692ae28fda9bfbde7c01b13bf5a03395-Abstract-Conference.html), [Pattern Shifting or Knowledge Losing?](https://aclanthology.org/2024.ccl-1.106/), [Spurious Forgetting](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a774503daed55eb53c634847ae071ec7-Abstract-Conference.html), and [Function Vectors](https://proceedings.iclr.cc/paper_files/paper/2025/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html).

5. **Target learning must be controlled.** A method that learns less will often appear to forget less. Equal epochs, steps, or examples are not enough. Comparisons should use target-retention curves or checkpoints matched within a pre-specified target-performance tolerance.

6. **Replay remains a strong baseline.** Small real-data replay, source-data injection, and synthetic replay frequently match or beat complicated methods. The relevant resource is not only stored examples: papers make different assumptions about access to labels, prompts, logits, base-model generations, activation statistics, or source corpora.

7. **There is no universal state-of-the-art method.** Static domain adaptation, sequential task learning, continual pretraining, modular routing, and model merging solve different problems. Their numbers cannot be placed in a single defensible leaderboard.

### Findings that remain uncertain or conflict

- **Model scale.** Some 1B-13B studies report larger raw drops for larger models; controlled scratch-model work finds smaller models forget more; recent paired-checkpoint studies often find larger models forget less. Parameter count is confounded with base competence, headroom, architecture, training duration, and data.
- **Best spectral subspace.** Principal, minor, and intermediate singular components have each been reported as preferable. Update magnitude can explain some apparent spectral effects.
- **Bottom-layer freezing.** It is highly effective in the spurious-forgetting setting, but other studies find naive freezing unstable or inferior to source-importance masks. Which parameters are frozen matters more than the fraction alone.
- **LoRA versus full fine-tuning at matched learning.** LoRA often retains more, but not always. Sequential LoRA can accumulate destructive directions and eventually forget more than full tuning in controlled encoder experiments.
- **Recovery versus relearning.** Prompt-only recovery is stronger evidence of retained access than gradient-based repair. A few fine-tuning steps can either unlock an old behavior or relearn it. Current terminology does not cleanly separate these cases.

### Main research trends

- 2023: document forgetting and build continual instruction benchmarks.
- 2024: characterize the learning-retention trade-off; distinguish task inference, topic/style shift, and factual change; revisit replay and loss geometry.
- 2025: move to item-level prediction, singular directions, token difficulty, function vectors, targeted rehearsal, and controlled LoRA/full-FT comparisons.
- 2026: expand spectral and subspace control, layer-aware regularization, denoising, realistic memory assumptions, large-scale post-training audits, and comparisons between supervised and reinforcement fine-tuning.

### Three best directions

1. **Damage-aware KL anchoring with recovery-aware evaluation.** Best overall balance of novelty, scientific value, and feasibility.
2. **Prompt-recoverable versus durable forgetting under matched LoRA and full fine-tuning.** Safest and most rigorous minimum viable project.
3. **A unified predictor study of LoRA forgetting using magnitude, effective rank, spectral intrusion, gradient conflict, and activation drift.** Most mechanistic and PhD-like, but technically riskier.

## Review methodology

### Search process

The review used a bounded systematic discovery pass followed by targeted snowballing and conflict searches.

- **Indexes and venues:** OpenAlex, arXiv, OpenReview, ICLR, ICML/PMLR, NeurIPS proceedings, ACL Anthology, AAAI proceedings, TMLR, ACM Computing Surveys, and official code/project pages.
- **Core query families:** "catastrophic forgetting large language model fine tuning," "continual learning large language models instruction tuning," "LoRA catastrophic forgetting," "parameter efficient fine tuning knowledge retention," "continual instruction tuning replay," "spurious forgetting language models," "knowledge distillation forgetting LLM," "spectral subspace LoRA forgetting," "gradient conflict continual fine tuning," and "adapter composition interference."
- **Time range:** primarily January 2023 through 4 September 2026. Earlier work was retained only for essential definitions, metrics, LoRA, continual-learning baselines, or replay foundations.
- **Discovery counts:** ten broad index queries returned 1,000 records. Title normalization and cross-query deduplication left 746 unique indexed candidates. The 180 highest-relevance records were retained for bounded title/abstract screening. Targeted venue, citation, and official-page searches were then used to correct metadata, add recent 2026 work, and resolve conflicts.
- **Close review:** 60 papers were inspected closely enough to record research question, model scale, data, tuning method, baselines, metrics, main result, limitations, code status, and approximate compute feasibility. The remaining 120 screened records were not promoted to the close-review set.

### Inclusion criteria

- Direct evidence about capability degradation, catastrophic forgetting, continual adaptation, or retention during language-model fine-tuning.
- LoRA/PEFT, full fine-tuning, continual instruction tuning, replay, regularization, gradient/subspace methods, routing, or merging with a clear retention question.
- Decoder-only evidence preferred. Encoder, multimodal, and classical continual-learning papers were included only when they supply a direct mechanism, metric, benchmark, or baseline relevant to LLM fine-tuning.
- Public paper or official proceedings record with verifiable metadata.

### Exclusion and downweighting criteria

- Duplicated arXiv and conference versions; the final publication was preferred.
- Vision-only continual learning without a transferable LLM mechanism.
- Ordinary fine-tuning efficiency or data-selection papers with no retention question.
- Unlearning papers unless they clarify access versus erasure.
- Unverifiable manuscripts claiming experiments on inaccessible closed-model weights.
- Very recent preprints were retained when directly relevant but labeled as unreviewed.

### Search limitations

This is a course-selection review, not a registered PRISMA meta-analysis. Search indexes lag some 2026 proceedings, venue metadata can change, and many papers omit seeds, exact compute, or comparable retention measurements. The numerical screening score was used only to bound human review; it is not a quality score. Search saturation was judged when additional focused queries repeated the same mechanism families and close neighbors. Absence from this corpus is not proof that a question is unstudied.

## Definitions and taxonomy

### What current papers call "forgetting"

The same label is used for several different events:

1. **Old-task catastrophic forgetting:** performance on an earlier explicitly learned task drops after a later task.
2. **General-capability degradation:** an untouched pretrained or instruction-tuned capability drops after specialization.
3. **Continual-pretraining retention loss:** source-distribution next-token loss increases while adapting to a new corpus.
4. **Alignment tax:** safety, helpfulness, instruction following, calibration, or diversity declines after another post-training stage.
5. **Behavioral interference:** the adapted policy favors the new behavior even though an older behavior can be elicited.
6. **Prompt-format sensitivity:** an answer is semantically available but not produced or extracted under one template.
7. **Genuine or durable loss:** degradation persists across valid elicitation formats, likelihood scoring, and modest recovery probes. This is stronger evidence, but even then "erasure" is too strong without mechanistic tests.

### Three-layer interpretation

For a practical project, observed loss should be decomposed into:

- **Evaluation artifact:** parser, exact-match, refusal, chat-template, or answer-format failure.
- **Access or alignment failure:** the capability is recovered by meaning-preserving prompts, few-shot context, or an inference-time intervention.
- **Residual capability loss:** the difference that persists across the pre-registered recovery battery.

The project should call the last category "residual" or "durable under our probes," not "proven knowledge erasure."

### Mechanism families

- Update capacity: trainable parameter count, LoRA rank, layer/module coverage.
- Update scale: learning rate, alpha, Frobenius norm, spectral norm, training duration.
- Update geometry: effective rank, singular-vector intrusion, overlap with pretrained subspaces, activation drift.
- Optimization: sharpness, high-loss tokens/batches, gradient conflict, checkpoint overshoot.
- Task relationship: semantic similarity, format similarity, function-vector similarity, source-target distribution shift.
- Parameter location: bottom versus top layers, attention versus MLP, source-important columns or weights.
- System isolation: separate adapters, identity routes, expert pools, architectural expansion.

### Mitigation families

- Real replay and source-data injection.
- Synthetic or self-generated replay.
- KL anchoring and self-distillation.
- Weight, curvature, or sharpness regularization.
- Orthogonal, null-space, and projected updates.
- Layer freezing and selective/sparse fine-tuning.
- Spectral clipping, denoising, and subspace selection.
- Model merging and task-vector pruning.
- Multiple adapters, routing, and mixture-of-experts.
- Data selection, curricula, and loss-adaptive optimization.

## Historical and thematic synthesis

### Foundations to 2022

Classical continual learning supplied the stability-plasticity framing, replay, EWC, GEM/gradient projection, and ACC/BWT/FWT metrics. LoRA introduced an efficient low-rank update parameterization. Language-model studies then began testing continual pretraining and instruction streams. [Fine-tuned Language Models Are Continual Learners](https://aclanthology.org/2022.emnlp-main.410/) showed that broad instruction tuning plus approximately 1% replay could be a strong baseline.

### 2023: benchmarks and isolation

[CITB](https://aclanthology.org/2023.findings-emnlp.633/) and [TRACE](https://arxiv.org/abs/2310.06762) made long instruction streams and broader retained abilities visible. [O-LoRA](https://aclanthology.org/2023.findings-emnlp.715/) used separate, orthogonal adapter subspaces. This wave asked whether existing continual-learning tools transfer to instruction-tuned language models.

### 2024: trade-offs, access, replay, and optimization

[LoRA Learns Less and Forgets Less](https://openreview.net/forum?id=aloEru2qCG) showed that the LoRA/full-FT comparison is a learning-retention curve rather than a binary win. [Implicit Inference](https://proceedings.iclr.cc/paper_files/paper/2024/hash/692ae28fda9bfbde7c01b13bf5a03395-Abstract-Conference.html), [Dissecting Learning and Forgetting](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c4de749a4bd8802f0b4033d09a7867db-Abstract-Conference.html), and [Pattern Shifting](https://aclanthology.org/2024.ccl-1.106/) separated task inference, topic/style priors, and factual changes. [Revisiting Catastrophic Forgetting](https://aclanthology.org/2024.findings-emnlp.249/) linked sharpness with forgetting. [SSR](https://aclanthology.org/2024.acl-long.77/) and [InsCL](https://aclanthology.org/2024.naacl-long.37/) made replay more data-aware or synthetic.

### 2025: item-level and geometric diagnosis

[Spurious Forgetting](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a774503daed55eb53c634847ae071ec7-Abstract-Conference.html) formalized an alignment-loss component. [LoRA vs Full Fine-Tuning](https://papers.nips.cc/paper_files/paper/2025/hash/ff541950d1e885af90f523571564a401-Abstract-Conference.html) identified intruder singular directions. [Low-Perplexity Token Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/027e86facfe7c1ea52ca1fca7bc1402b-Abstract-Conference.html) implicated hard target tokens. [Low-Rank Example Associations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/06872e1e6d11baf2ae27285c50132f4f-Abstract-Conference.html) predicted which upstream examples would be forgotten and used targeted replay. CLoRA, GORP, N-LoRA, CaLoRA, SLIM, and DEAL explored different constraints, projections, and routing mechanisms.

### 2026: subspace control, realistic resources, and post-training maps

ACL/AAAI work includes SLoRA, ASO-LoRA, SpaRTA, SCLoRA, OPLoRA, Source-Shielded Updates, and L2-LoRA. These methods define protected knowledge using different and sometimes contradictory geometries. [Mapping Post-Training Forgetting at Scale](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bc1c5e5fb8ed1ef9b9b5abced2022e40-Abstract-Conference.html) emphasizes item-level transitions. [Forget Forgetting](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html) argues that storage may be cheaper than GPU time, making simple replay more realistic. Another emerging line compares SFT with reinforcement fine-tuning, often finding lower distribution drift for on-policy RL, but that is a larger project than this team needs.

## State-of-the-art comparison by experimental regime

### Regime A: one-stage specialization with general retention

Common setup: fine-tune on math, code, medicine, or instruction data; evaluate an untouched general suite.

- Full FT often reaches higher target performance but can spend more general capability.
- Standard LoRA often retains more, partly because its update is constrained.
- Recent spectral, orthogonal, and layer-aware methods report improved Pareto points, but each uses different models, prompts, retention sets, ranks, and stopping rules.
- The 2026 [PEFT-Arena](https://arxiv.org/abs/2605.28819) preprint is the broadest direct PEFT comparison found, but it covers two model families and two target domains and remains unreviewed.

**SOTA judgment:** no defensible universal winner. OFT, KeepLoRA/VeRA, OPLoRA, SCLoRA, L2-LoRA, FINCH, SAM, and low-perplexity-token methods are strong within their own protocols. They should be treated as candidate baselines, not a single ranked list.

### Regime B: sequential continual instruction tuning

Common setup: train through 5-16 tasks and measure old-task average, BWT/FWT, and sometimes general capability.

- Replay methods such as SSR and InsCL are strong and conceptually simple.
- Orthogonal/subspace methods such as O-LoRA, GORP, CLoRA, ASO-LoRA, SLoRA, SpaRTA, and DEAL report improvements.
- Modular methods such as SEE and MoCL can approach zero forgetting by retaining separate modules and routing inputs.

**SOTA judgment:** SLoRA, SpaRTA, ASO-LoRA, DEAL, and KPIG are among the strongest recent reported systems, but comparisons are not controlled across storage, task identifiers, number of adapters, old data, teacher calls, and compute. Random replay remains mandatory.

### Regime C: continual pretraining and domain adaptation

- Source-data injection and replay plus learning-rate rewarming are consistently strong.
- Scaling-law work suggests small source fractions can preserve source loss in controlled small-model regimes.
- These conclusions cannot be transferred directly to instruction-following accuracy because next-token source loss measures a different object.

**SOTA judgment:** replay/injection plus a correct optimizer schedule is the strongest practical baseline. Paper-scale token budgets are inappropriate for this course.

### Regime D: routing, architectural expansion, and merging

- Separate experts, identity routes, and growing adapters preserve the base by isolation.
- They change the resource question: total storage, active parameters, task/OOD routing accuracy, latency, and task-ID assumptions must be reported.
- Merging is cheap after experts exist, but the 2026 mapping study finds that public model merging does not reliably eliminate item-level forgetting.

**SOTA judgment:** these are systems solutions, not proof that a single adapted model retained its knowledge internally.

## Evaluation audit and recommended protocol

### Classical task-stream metrics

Let R[t,i] be the score on task i after training through task t.

- Final average accuracy: ACC = mean_i R[T,i]
- Backward transfer: BWT = mean_i<T (R[T,i] - R[i,i])
- Average forgetting: FM = mean_i<T (max_k<T R[k,i] - R[T,i])

BWT and FM are not interchangeable. BWT compares with performance just after learning each task. FM compares with the best historical checkpoint.

### Minimum credible package for a small-LLM project

1. **Target learning:** target exact match/accuracy plus target loss.
2. **Base-relative retention:** absolute and relative change for every retained capability.
3. **Item transitions:** correct-to-incorrect forgetting and incorrect-to-correct backward transfer.
4. **Generated and likelihood scoring:** standard generation plus forced-choice log-likelihood for multiple-choice tasks.
5. **Format compliance:** parse failure, refusal, wrong answer schema, and over-generation rates.
6. **Prompt battery:** at least five pre-registered meaning-preserving templates; report mean, median, and worst case, not the best template alone.
7. **Recovery battery:** zero-shot template changes, fixed few-shot demonstrations, and an optional tiny alignment-only repair. Label gradient-based repair as ambiguous between recovery and relearning.
8. **Target-matched comparison:** compare checkpoints within a pre-specified target-performance tolerance or interpolate along the learning-retention frontier.
9. **Mechanism diagnostics when claimed:** update Frobenius norm, spectral norm, effective rank, layer/module coverage, and at least one activation or gradient measure.
10. **Resources:** training tokens, effective batch size, number of forwards/backwards, peak memory, wall time, and stored replay/adapter size.

### Statistical requirements

- Use at least two training seeds for every central comparison and three where budget permits.
- Use paired bootstrap confidence intervals over evaluation items because the same examples are scored before and after tuning.
- Treat prompt templates as repeated measurements rather than choosing the most favorable one.
- If task order is studied, use at least three pre-registered orders or explicitly label order evidence as exploratory.
- When fitting predictors from many checkpoints, split or cross-validate by training run, not randomly by checkpoint, to avoid leakage.
- Report null results and effect intervals. Do not use overlapping bars as the only test.

### Suitable public data for 0.5B-3B models

- **Target math:** GSM8K subset or MetaMathQA subset. GSM8K offers clean numeric exact match but is narrow.
- **Target code:** MBPP or a small code-instruction subset. Evaluation cost is higher because execution is required.
- **Sequential tasks:** a three-task mini-TRACE or three diverse SuperNI categories. Avoid the full long stream initially.
- **Knowledge/reasoning retention:** MMLU subject subsets, ARC-Challenge, BoolQ, OpenBookQA, HellaSwag.
- **Instruction following:** a compact deterministic IFEval subset.
- **General text retention:** WikiText-103 or C4 held-out next-token loss, used as a distributional metric rather than a direct capability score.
- **Unlabeled anchor pool:** a filtered public instruction-prompt pool such as Dolly-15k prompts. Remove evaluation overlap and never use evaluation questions as anchors.

### Leakage and contamination controls

- Keep target train, anchor candidate pool, alignment-repair examples, validation, and final evaluation disjoint.
- Search for exact and near-duplicate questions across pools.
- Do not tune anchor selection thresholds on the final retention set.
- Report model checkpoint and dataset versions.
- Acknowledge that pretrained models may already have seen public benchmarks; the study measures retained behavior, not clean acquisition of unknown knowledge.

## Research-gap map

| Gap | Classification | Confidence | Evidence and caution |
|---|---|---:|---|
| Recovery-aware LoRA versus full FT at matched target gain | Studied only in intersecting but separate settings | Medium-high | LoRA/full FT and spurious forgetting are each mature; their controlled intersection with multiple recovery probes is limited. |
| Damage-aware selection of unlabeled KL anchors | Limited / missing direct control | Medium | Damage-aware labeled replay, KL anchoring, and example prediction exist separately. The exact fixed-budget combination was not found. |
| Unified checkpoint-level predictor of forgetting | Missing comparison/control | High | Magnitude, intruders, effective rank, gradient conflict, sharpness, and activation drift are usually evaluated in separate papers. |
| Prompt-robust forgetting metric | Missing evaluation standard | High | Prompt sensitivity and access failure are established, but most forgetting papers still use one template. |
| Active versus disabled versus merged adapters | Studied only in limited settings | Medium | Disabling trivially restores the base; few studies quantify interference while active versus after merging at matched target gain. |
| When orthogonality helps versus blocks transfer | Contradictory evidence | High | Strict orthogonality protects dissimilar tasks but may prevent beneficial transfer; task-similarity measures disagree. |
| Principal versus minor versus intermediate spectral directions | Contradictory evidence | High | Recent papers recommend all three under different setups; magnitude and task distance are confounds. |
| Small-model generality of 7B-32B methods | Missing scale validation | High | Many recent methods use 7B+ and no sub-3B validation. |
| Long-horizon adapter composition with fair resource accounting | Limited systems evaluation | Medium-high | Routing methods often omit total storage or compare against non-growing baselines. |
| Proving genuine knowledge erasure | Promising but computationally and conceptually difficult | High | Behavioral non-recovery cannot prove absence; causal representation tests are expensive and theory remains incomplete. |

## Verdict on the team's starting ideas

| Starting idea | Verdict | How to use it |
|---|---|---|
| Plain LoRA versus full FT | Abandon as a novelty claim | Keep both as baselines; match target learning and tune learning rates. |
| LoRA/full FT plus KD anchoring | Plain version already occupied | Make anchor selection and recovery-aware evaluation the contribution. |
| Disentangle rank, frozen backbone, and update magnitude | Too crowded as stated | Replace with a unified predictor study using identical checkpoints and matched target gain. |
| Recoverable/spurious versus genuine forgetting | Pursue | Use precise operational categories and avoid claiming proven erasure. |
| Layer-wise or adaptive LoRA rank | Crowded | Only pursue if rank is allocated by a tested predictive signal under a fixed parameter budget. |
| Predict forgetting from gradient/subspace conflict | Pursue as diagnosis | Compare against magnitude, spectrum, task loss, and activation drift; simple gradient cosine may be weak. |
| Adapter composition and interference | Viable systems project | Account for storage, routing, task IDs, latency, and base fall-through. |

## Eleven realistic project ideas

Scores use 1-5. For complexity, compute, and inconclusive-result risk, 1 is better/easier and 5 is worse/harder.

| Rank | Project direction | Novelty | Scientific value | Feasibility | Complexity | Compute | Data/code | Null-risk | PhD value |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Damage-aware KL anchor selection plus recovery-aware evaluation | 4 | 5 | 4 | 3 | 3 | 5 | 3 | 5 |
| 2 | Prompt-recoverable versus durable forgetting under target-matched LoRA/full FT | 4 | 5 | 5 | 2 | 3 | 5 | 2 | 5 |
| 3 | Unified geometric predictor of LoRA forgetting | 4 | 5 | 3 | 4 | 3 | 4 | 4 | 5 |
| 4 | Prompt-robust forgetting benchmark for small LLMs | 3 | 4 | 5 | 2 | 1 | 5 | 2 | 4 |
| 5 | Active, disabled, scaled, and merged adapter interference | 4 | 4 | 5 | 2 | 2 | 5 | 2 | 4 |
| 6 | Damage-aware labeled replay for decoder-only instruction tuning | 3 | 4 | 5 | 3 | 2 | 5 | 2 | 4 |
| 7 | Optimizer reality check: standard SFT versus SAM, FINCH, replay, and KL | 3 | 4 | 4 | 3 | 3 | 4 | 3 | 4 |
| 8 | Task-distance map for when orthogonal LoRA helps or hurts transfer | 4 | 5 | 3 | 4 | 3 | 4 | 4 | 5 |
| 9 | Adapter composition order and routing interference | 3 | 4 | 4 | 3 | 2 | 4 | 3 | 4 |
| 10 | Real versus self-generated replay under equal token and inference budgets | 2 | 4 | 3 | 3 | 3 | 4 | 3 | 4 |
| 11 | Source-importance freezing versus random, bottom-layer, and magnitude masks | 2 | 3 | 4 | 3 | 3 | 4 | 3 | 3 |

## Proposal 1: Damage-aware KL anchoring

### Candidate titles

- Replay or Re-alignment? Damage-Aware KL Anchoring for Small Language Models
- Which Prompts Should a Fine-Tuned Model Remember?
- A Small Anchor Budget Goes a Long Way: Vulnerability-Guided Retention in LoRA

### Research question

Given a fixed number of unlabeled anchor tokens and equal target-task training, does selecting prompts by early base-to-adapted distribution drift preserve more general capability than random or diversity-only KL anchoring? Does it reduce residual forgetting after recovery controls?

### Falsifiable hypotheses

- **H1:** Damage-selected KL anchors produce lower general-capability loss than random anchors at the same anchor-token budget and target score.
- **H2:** Diversity-filtered damage anchors outperform simply taking the highest-drift prompts because redundant anchors waste budget.
- **H3:** KL anchoring improves forced-choice likelihood and multi-template robustness, not only generated exact match.
- **H4:** Some of the raw gain from anchoring disappears after prompt/few-shot recovery, showing that the method preserves alignment as well as durable capability.

### Relationship to prior work and novelty

The project combines four established threads: KL/self-distillation, replay selection, example-level forgetting prediction, and spurious forgetting. [mix-cd](https://aclanthology.org/2025.findings-naacl.138/) selects labeled examples that become collateral damage, but not decoder-only unlabeled KL anchors. [Low-Rank Example Associations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/06872e1e6d11baf2ae27285c50132f4f-Abstract-Conference.html) predicts forgotten examples but does not provide this recovery-aware, fixed-anchor-budget LoRA protocol. [Context-Free Synthetic Data](https://arxiv.org/abs/2505.13811) studies which generated inputs approximate KL, but not pilot-damage selection. The claim should therefore be an **extension and controlled intersection**, not a claim that KL or targeted replay is new.

### Models and data

- Primary: Qwen2.5-0.5B-Instruct.
- Confirmation: Qwen2.5-1.5B-Instruct if 24-48GB is available.
- Target: 2,000-4,000 GSM8K or MetaMathQA training examples; hold out validation and test.
- Anchor candidate pool: 2,000 filtered Dolly-15k or another general instruction-prompt pool, used without labels.
- Retention: fixed subsets of MMLU non-math subjects, ARC-Challenge, BoolQ, OpenBookQA, IFEval, and WikiText/C4 loss.
- Recovery: five equivalent templates per task, fixed few-shot examples from development splits, and an optional 32-example format-only repair set.

### Baselines and core matrix

| Condition | Selection | Retention objective | Role |
|---|---|---|---|
| Base | None | None | Untouched reference |
| Task-only LoRA | None | None | Main forgetting baseline |
| Random-KL | Random prompts | KL to base | Plain anchoring baseline |
| Diversity-KL | Embedding clusters | KL to base | Coverage baseline |
| Damage-KL | Highest pilot drift | KL to base | Proposed selector |
| Diverse-Damage-KL | Drift then clustering | KL to base | Proposed selector plus redundancy control |
| Random labeled replay | Random old/general examples | Cross-entropy | Replay baseline if labels are permitted |
| Full FT task-only | None | None | Optional 0.5B baseline |

Main design: five central LoRA conditions x two seeds = 10 runs, plus two task-only full-FT seeds and three 1.5B confirmation runs = 15 essential runs. Select checkpoints by target validation score within a pre-registered tolerance, not final epoch alone.

### Implementation detail

1. Train a short warm-up LoRA on 5%-10% of target steps.
2. For each anchor candidate, compute KL or logit drift between the base and warm-up model on response positions. If response tokens are absent, generate a short base completion and score that fixed continuation.
3. Select the top drift prompts; for the diverse version, cluster prompt embeddings and choose high-drift examples across clusters.
4. Restart from the same base and train the main run. Alternate target batches with anchor batches. Compute the teacher distribution by temporarily disabling the LoRA adapter, or cache a top-k teacher distribution to reduce cost.
5. Keep total anchor tokens, target tokens, optimizer steps, and teacher forward passes comparable.

### Metrics

- GSM8K numeric exact match and target negative log-likelihood.
- Per-task absolute and relative retention drop.
- Item-level correct-to-incorrect and incorrect-to-correct transitions.
- Forced-choice log-likelihood accuracy.
- Template mean, median, and worst-case accuracy; format compliance.
- Prompt-recoverable drop and residual drop under the fixed recovery battery.
- KL anchor efficiency: retention gain per 100,000 anchor tokens and per GPU-hour.

### Ablations

- Anchor budgets: 32, 128, 256 prompts.
- Drift score: sequence KL, answer-token margin change, or loss change.
- Top-drift versus drift-plus-diversity.
- KL direction and temperature.
- Early warm-up checkpoint used for selection.
- Attention-only versus all-linear LoRA for the winning selector.

### Statistical analysis

Use paired bootstrap intervals over evaluation items and show individual seed points. For the anchor-budget curve, fit a simple monotonic or log-budget trend only if supported. Pre-register the primary comparison as Diverse-Damage-KL versus Random-KL at the middle budget. Correct secondary pairwise tests or report them descriptively.

### Expected figures and tables

- Target gain versus raw and residual forgetting Pareto plots.
- Retention gain per anchor token for random, diversity, damage, and diverse-damage selection.
- Item transition chart: retained, forgotten, recovered by prompt, recovered by few-shot, residual.
- Table of target score, retained capability, format failures, wall time, and storage.

### Interpretation

- **Positive:** damage-aware anchoring improves raw and residual retention at matched target learning. This supports targeted protection of vulnerable behavior.
- **Only raw improves:** the selector mainly preserves task alignment or format access, still a useful practical result.
- **Null:** random prompts are sufficient under this budget, or pilot damage is too unstable. This is publishable as a negative result if confidence intervals are tight.
- **Negative:** high-drift prompts may be outliers or may overconstrain adaptation. Analyze redundancy and task similarity.

### Compute plan

- **16GB:** Qwen2.5-0.5B, LoRA/QLoRA, sequence length 512, gradient checkpointing; 4-5 central conditions and two seeds. Omit full FT if unstable.
- **24GB:** 0.5B full FT plus 1.5B LoRA confirmation; 5-6 conditions.
- **48GB:** 1.5B full FT or 3B PEFT; three seeds for the primary comparison.

### Eight-week plan and team split

- Week 1: reproduce base evaluation; freeze prompts, splits, and metrics.
- Week 2: stable task-only LoRA/full-FT baselines; checkpointing and logging.
- Week 3: random KL and random replay.
- Week 4: pilot drift scoring and damage selection.
- Week 5: central two-seed matrix and target matching.
- Week 6: recovery battery and one budget ablation.
- Week 7: statistics, error analysis, confirmation run.
- Week 8: report, presentation, release and reproducibility check.

Suggested ownership: member A training infrastructure; member B anchor scoring/selection; member C evaluation/recovery; member D statistics, experiment tracking and qualitative analysis. Everyone owns at least one reproduction check.

### Minimum viable and fallback

MVP: one 0.5B model, one math target, task-only LoRA, Random-KL and Damage-KL, two seeds, three retention tasks, forced-choice scoring and five prompt templates. Fallback if KL is too expensive: use logit matching only on the teacher top-k tokens or replace KL with base-generated pseudo-label cross-entropy. Fallback if no forgetting appears: increase target steps only until a measurable but non-collapsed target gain is reached, or use a more format-distant target task.

## Proposal 2: Recoverable versus durable forgetting

### Candidate titles

- Forgotten or Misaligned? A Target-Matched Comparison of LoRA and Full Fine-Tuning
- Does LoRA Preserve Knowledge or Merely Access to It?
- Beyond Benchmark Drop: Recovery Profiles of Fine-Tuned Small Language Models

### Research question

At equal target-task improvement, how much observed forgetting under LoRA and full fine-tuning is attributable to output format, prompt sensitivity, task alignment, or residual capability loss?

### Hypotheses

- **H1:** A nontrivial fraction of standard benchmark drop is recovered by equivalent prompts or likelihood scoring.
- **H2:** LoRA has lower raw forgetting than full FT, but the difference shrinks after target matching.
- **H3:** Low-rank LoRA produces more prompt-sensitive access failures, while full FT produces a larger residual loss. This is deliberately falsifiable; the opposite result is plausible.
- **H4:** Output-format distance predicts recoverable loss better than semantic task distance.

### Prior work and novelty

The individual ingredients are established, but the field lacks a standard recovery battery applied to target-matched LoRA/full-FT checkpoints on current small decoder models. This is primarily a **measurement and controlled-evaluation contribution**. It is safer than inventing a method and remains useful if LoRA and full FT are indistinguishable.

### Models, data, and experiment matrix

- Qwen2.5-0.5B-Instruct primary; optional 1.5B confirmation.
- Target A: GSM8K/MetaMathQA; target B: a format-distant classification or code-instruction subset.
- Methods: full FT; LoRA ranks 8 and 64; untouched base.
- Two seeds for 3 methods x 2 targets = 12 runs. Save 4-6 checkpoints per run and compare target-matched points.
- Retention: MMLU non-target subjects, ARC-C, BoolQ, IFEval, and general-text loss.

### Recovery battery

1. Standard generation and official template.
2. Five meaning-preserving prompt templates with matched base-model evaluation.
3. Constrained/forced-choice log-likelihood.
4. Fixed 3-shot demonstrations.
5. Optional 32-example format-only repair on a separate adapter.

Report prompt-recoverable, few-shot-recoverable, intervention-recoverable, and residual loss. Do not call the residual "erased knowledge."

### Metrics, ablations, and statistics

Use target gain, raw/relative drop, item transitions, extraction failure, template variance, and recovery fraction. Ablate rank, target format, and checkpoint depth. Use paired bootstrap intervals and seed points. Test whether format distance and update statistics predict each loss component.

### Figures and interpretations

- Stacked decomposition of raw drop into parser/format, prompt-recoverable, few-shot-recoverable, and residual.
- Target-retention frontier for each method.
- Item-level confusion/transition table.

A null method difference is still valuable if the recovery protocol shows that standard single-template evaluation overstates forgetting. If no recovery occurs, that is evidence that the chosen adaptation produces durable loss under a strong set of probes.

### Compute, schedule, and division

The 0.5B 12-run design fits 16-24GB. Full 1.5B FT is reserved for 48GB. Weeks 1-2 build evaluation; weeks 3-4 run target A; week 5 runs target B; week 6 recovery; week 7 statistics; week 8 reporting. Assign training, prompt/evaluation, likelihood/recovery, and statistical/error-analysis leads.

### MVP and fallback

MVP: one target, full FT and two LoRA ranks, two seeds, three retained tasks, standard/forced-choice/five-template evaluation. If full FT is unstable on 16GB, use Qwen2.5-0.5B with an 8-bit optimizer or compare LoRA with partial full tuning and state the limitation.

## Proposal 3: Unified predictors of LoRA forgetting

### Candidate titles

- Beyond Rank: What Predicts Catastrophic Forgetting in LoRA?
- Magnitude, Intruders, or Interference? A Controlled LoRA Mechanism Study
- One Set of Checkpoints, Five Explanations of Forgetting

### Research question

After controlling for target learning, which checkpoint-level signal best predicts retained-capability loss: update magnitude, effective rank, intruder singular directions, pretrained-subspace overlap, activation drift, or gradient conflict?

### Hypotheses

- **H1:** Rank alone loses predictive power after controlling for target score and update norm.
- **H2:** Capability-conditioned activation drift predicts forgetting more reliably than raw parameter movement.
- **H3:** Intruder count predicts retention in low-rank/high-learning-rate settings but is not universal across modules and tasks.
- **H4:** Simple gradient cosine similarity is weaker than activation drift or pilot item loss for example-level prediction.

### Prior work and novelty

Every candidate signal has prior literature. The contribution is the **controlled comparison on identical checkpoints**, not a new metric. This directly addresses conflicting explanations from LoRA/full-FT studies, magnitude-controlled adaptation, sharpness, spectral clipping, PEFT-Arena, and gradient/subspace methods.

### Models, data, and core matrix

- Qwen2.5-0.5B or 1.5B.
- Main target: GSM8K subset; validation target: small code or instruction-format task.
- LoRA ranks 4, 16, and 64.
- Attention-only versus all-linear targeting.
- Standard versus update-norm-rescaled configuration for selected ranks.
- Task-only full FT reference if feasible.
- Central 6 configurations x 2 seeds = 12 runs; three second-task confirmations and two intervention runs keep the total under 18.

Save checkpoints at fixed target-score intervals. Measure per layer: Frobenius norm, spectral norm, stable/effective rank, singular-vector overlap, intruder count, activation CKA or relative activation change on a fixed retention set, and target-versus-retention gradient cosine on a small batch.

### Analysis

- Compare correlations after residualizing on target score and training step.
- Use leave-one-run-out or leave-one-configuration-out cross-validation.
- Bootstrap evaluation items and cluster checkpoint observations by training run.
- Avoid causal wording for predictors.
- Stretch intervention: scale or clip only layers exceeding the best validated risk signal, then re-evaluate target/retention.

### Figures

- Target-matched forgetting by rank, module set, and update norm.
- Predictor correlation matrix and cross-validated prediction error.
- Layer-depth heat maps for update and activation drift.
- Pre/post intervention Pareto shift.

### Interpretations

If one signal generalizes across ranks and tasks, it motivates future adaptive protection. If magnitude dominates, several geometric claims may be confounded. If no signal generalizes, forgetting may be capability-conditioned and require example-level rather than weight-only diagnostics.

### Compute, schedule, and team split

This is feasible on 24GB with a 0.5B-1.5B model; full 1.5B FT and repeated full SVD are best on 48GB. Use randomized/truncated SVD and selected layers on 16GB. Weeks 1-2 establish training/eval; weeks 3-4 generate checkpoints; week 5 compute geometry; week 6 fit predictors; week 7 intervention/validation; week 8 report. Assign training, spectral analysis, activation/gradient diagnostics, and statistics/reproducibility leads.

### MVP and fallback

MVP: one target, three ranks, two module sets, two seeds; update norm, effective rank, singular-vector overlap, and one activation-drift measure. Drop gradient cosine first if implementation is unstable. If forgetting is too small, use longer tuning but retain target-matched checkpoints rather than comparing only endpoints.

## Recommended final choice

Choose **Proposal 1** if the team wants a method plus a diagnostic contribution. Choose **Proposal 2** if reliability and finishing well matter most. Choose **Proposal 3** if the team has one strong systems/linear-algebra member and wants a more mechanistic PhD-style project.

The practical decision rule is:

- One 16GB GPU and heavy coursework: Proposal 2 MVP.
- Reliable 24GB access and four members: Proposal 1.
- 48GB access, strong implementation experience, and appetite for analysis risk: Proposal 3.

## Selected evidence notes

The companion workbook contains the full 60-paper evidence matrix and the 180-record screening log. The following papers are especially important for the project decision:

1. [LoRA Learns Less and Forgets Less](https://openreview.net/forum?id=aloEru2qCG) - establishes the basic LoRA/full-FT trade-off.
2. [LoRA vs Full Fine-Tuning: An Illusion of Equivalence](https://papers.nips.cc/paper_files/paper/2025/hash/ff541950d1e885af90f523571564a401-Abstract-Conference.html) - identifies intruder singular directions and shows hyperparameters can reverse retention.
3. [Spurious Forgetting](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a774503daed55eb53c634847ae071ec7-Abstract-Conference.html) - separates alignment loss from underlying knowledge in controlled settings.
4. [Implicit Inference](https://proceedings.iclr.cc/paper_files/paper/2024/hash/692ae28fda9bfbde7c01b13bf5a03395-Abstract-Conference.html) - demonstrates prompt-based recovery.
5. [Mapping Post-Training Forgetting at Scale](https://proceedings.iclr.cc/paper_files/paper/2026/hash/bc1c5e5fb8ed1ef9b9b5abced2022e40-Abstract-Conference.html) - supplies item-level forgetting and backward-transfer analysis.
6. [Revisiting Catastrophic Forgetting](https://aclanthology.org/2024.findings-emnlp.249/) - connects sharpness and forgetting.
7. [Scaling Laws with Pretraining Data Injection](https://proceedings.mlr.press/v267/bethune25a.html) - shows the strength of small source-data injection in controlled continual pretraining.
8. [Self-Synthesized Rehearsal](https://aclanthology.org/2024.acl-long.77/) - strong synthetic replay baseline.
9. [An Efficient Rehearsal Scheme](https://aclanthology.org/2025.findings-naacl.138/) - closest clean precedent for damage-aware sample selection.
10. [Low-Rank Example Associations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/06872e1e6d11baf2ae27285c50132f4f-Abstract-Conference.html) - predicts forgotten examples and weakens novelty claims around simple gradient similarity.
11. [Low-Perplexity Token Learning](https://proceedings.neurips.cc/paper_files/paper/2025/hash/027e86facfe7c1ea52ca1fca7bc1402b-Abstract-Conference.html) - shows data/token difficulty can drive forgetting.
12. [PEFT-Arena](https://arxiv.org/abs/2605.28819) - recent broad PEFT stability-plasticity comparison; preprint status matters.
13. [SLoRA](https://aclanthology.org/2026.acl-long.247/) - recent subspace-denoising approach for sequential LoRA.
14. [SCLoRA](https://aclanthology.org/2026.acl-long.1179/) - spectral-clipping approach.
15. [OPLoRA](https://ojs.aaai.org/index.php/AAAI/article/view/40703) - principal-subspace protection.
16. [ASO-LoRA](https://aclanthology.org/2026.acl-long.842/) - soft task-similarity-aware orthogonality.
17. [Source-Shielded Updates](https://aclanthology.org/2026.acl-long.865/) - source-importance selective freezing.
18. [FINCH](https://arxiv.org/abs/2605.20005) - simple loss-adaptive learning-rate proposal; promising but unreviewed.
19. [Forget Forgetting](https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html) - argues for realistic memory-versus-compute assumptions.
20. [H-LoRA](https://www.techscience.com/cmc/v88n1/67336/html) - directly relevant sub-1B high-rank study, but its recovery claim is weaker than the proposed operational recovery battery.

## Final cautions

- Do not claim novelty before one final focused search in the week the proposal is submitted.
- Do not report a single aggregate MMLU drop as "knowledge erased."
- Do not compare methods at equal steps only.
- Do not give modular methods a free storage or routing advantage.
- Do not spend the project implementing seven complex methods. A clean baseline, one contribution, and a strong evaluation are more research-worthy than a large but under-controlled matrix.
- A well-powered null result with released code and a recovery-aware protocol is a valid research outcome for this course.
