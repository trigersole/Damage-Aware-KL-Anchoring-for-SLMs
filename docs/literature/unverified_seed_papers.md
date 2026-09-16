# Catastrophic Forgetting in LLMs — Curated Paper List

> **Verification warning:** This file was recovered from an earlier seed list and may contain incomplete, outdated or inaccurate metadata. It is background material only. Verify every reference against a primary source before citing it. The reviewed evidence is in `llm_forgetting_sota_report_source.md`, `llm_forgetting_sota_evidence.xlsx`, and `screened_corpus.csv`.

A reading list organized by theme, for a course project on catastrophic forgetting (CF) and its mitigation. Roughly ordered from foundational → empirical characterization → mitigation methods → recent 2025/2026 work.

---

## 1. Foundations (pre-LLM, but essential background)

- **McCloskey & Cohen (1989)** — *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.* The paper that coined the phenomenon. Good for framing your intro/motivation section.
- **Kirkpatrick et al. (2017)** — *Overcoming Catastrophic Forgetting in Neural Networks (EWC).* PNAS.
  https://www.pnas.org/doi/10.1073/pnas.1611835114
  The classic regularization-based approach (Elastic Weight Consolidation) — penalizes changes to weights important for prior tasks, using the Fisher information matrix. Almost every LLM-era mitigation paper cites this as a baseline.
- **Lopez-Paz & Ranzato (2017)** — *Gradient Episodic Memory for Continual Learning (GEM).* NeurIPS. Classic memory/replay-based approach, often cited alongside EWC as the two canonical CL baselines.

## 2. Empirical characterization of CF in LLMs

- **Luo et al. (2023)** — *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning.*
  https://arxiv.org/abs/2308.08747
  Foundational LLM-specific empirical paper: shows CF is pervasive across 1B–7B models, gets worse with scale, and that decoder-only models (BLOOMZ) forget less than encoder-decoder (mT0). Also finds general instruction tuning helps.
- **Kotha et al. (2024)** — referenced widely re: fine-tuning risk of forgetting pretraining-scale capabilities (cited in survey below).
- **Zhao et al. (2023)** — *Speciality vs Generality: An Empirical Study on Catastrophic Forgetting in Fine-tuning Foundation Models.* arXiv:2309.06256.
- **Shuttleworth et al. (2025)** and **Biderman et al. (2024)** — comparative studies on whether LoRA forgets less than full fine-tuning (referenced in "Mitigating Forgetting in Low Rank Adaptation," below) — useful for a PEFT-vs-full-FT forgetting comparison angle.
- **"Chained Tuning Leads to Biased Forgetting"** (2024)
  https://arxiv.org/pdf/2412.16469
  Connects CF research to safety evaluations — relevant if your project has a safety/alignment angle.

## 3. Surveys (read these first for structure/taxonomy)

- **Continual Learning of Large Language Models: A Comprehensive Survey** — ACM Computing Surveys, 2025.
  https://dl.acm.org/doi/10.1145/3735633
  Broad taxonomy of CL for LLMs across pretraining, fine-tuning, and alignment stages — good for structuring a literature review.
- **Continual Learning in Large Language Models: Methods, Challenges, and Opportunities** (2026)
  https://arxiv.org/html/2603.12658v1
  Newer survey, categorizes by training stage (pretraining/fine-tuning/alignment); useful as a complementary structure.
- **AI Safety in Generative AI Large Language Models: A Survey** (2024) — has a dedicated CF section connecting forgetting to safety risk.
  https://arxiv.org/pdf/2407.18369

## 4. Mitigation: Regularization-based

- **Kirkpatrick et al. (2017)** — EWC (see above; still the reference baseline).
- **Li, Ding, Fang & Tao (2024)** — *Revisiting Catastrophic Forgetting in Large Language Model Tuning.* EMNLP Findings 2024.
  https://aclanthology.org/2024.findings-emnlp.249/
  Links CF severity to *sharpness of the loss landscape*; proposes sharpness-aware minimization (SAM) as a mitigation that complements other anti-forgetting strategies. Nice mechanistic angle if you want to go beyond "just apply method X."

## 5. Mitigation: Replay / Rehearsal-based

- **Scialom et al. (2022)**, **Mok et al. (2023)** — foundational rehearsal-based CL papers for LLMs (cited across many of the papers below; worth tracking down directly if replay is your focus).
- **Huang et al. (2024)** — *Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal (SSR).* ACL 2024.
  https://arxiv.org/pdf/2403.01244
  When you don't have access to original training data, generate synthetic rehearsal examples via in-context learning + self-refinement. Practical and widely cited.
- **Wu et al.** — *InsCL: A Data-efficient Continual Learning Paradigm for Fine-tuning LLMs with Instructions.*
  https://arxiv.org/pdf/2403.11435
  Instruction-aware replay strategy — replay sampling driven by instruction diversity rather than random sampling.
- **SERS — Self-Evolving Pseudo-Rehearsal for Catastrophic Forgetting with Task Similarity** (2025/2026, OpenReview)
  Decouples pseudo-input synthesis from label generation; addresses limitations of naive self-synthesis rehearsal.
- **Joint Flashback Adaptation for Forgetting-Resistant Instruction Tuning** (2025)
  https://arxiv.org/pdf/2505.15467
  Good related-work section summarizing replay vs. regularization vs. PEFT vs. adaptation-based approaches — useful as a mini-survey too.

## 6. Mitigation: Parameter-efficient fine-tuning (PEFT/LoRA) angle

This is a very active sub-area — LoRA constrains updates to a low-rank subspace, which changes (and can worsen, in some regimes) the forgetting profile relative to full fine-tuning.

- **On Catastrophic Forgetting in Low-Rank Decomposition-Based PEFT** (2026)
  https://arxiv.org/pdf/2603.09684
  Key finding: LoRA's low-rank constraint forces different tasks to share directions, increasing interference at low rank; full fine-tuning's unconstrained updates can actually reduce interference by using distinct directions.
- **Koubbi, Hernandez & Boussard — Understanding Catastrophic Forgetting in LoRA via Mean-Field Attention Dynamics** (2024)
  https://arxiv.org/pdf/2402.15415
  Theoretical/PDE-based analysis of LoRA forgetting as a phase transition — good if you want a theory-flavored angle.
- **CURLoRA: Stable LLM Continual Fine-Tuning and Catastrophic Forgetting Mitigation** (2024)
  https://arxiv.org/pdf/2408.14572 (code: https://github.com/mnoorfawi/curlora)
  Uses CUR matrix decomposition instead of standard LoRA factorization; implicit regularization via inverted-probability sampling.
- **O-LoRA experimental study** — *Mitigating Catastrophic Forgetting in Fine-Tuned LLMs: An Experimental Study of LoRA and O-LoRA* (2026)
  Orthogonal LoRA subspaces across tasks; empirically shown to help but hyperparameter-sensitive.
- **OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting** (2025)
  https://arxiv.org/pdf/2510.13003
  Projects LoRA updates orthogonal to top singular directions of the pretrained weight matrix to reduce interference with prior knowledge.
- **Mitigating Forgetting in Low Rank Adaptation** (2025/2026)
  https://arxiv.org/pdf/2512.17720
  Bayesian-inference-flavored approach to the LoRA learning/forgetting trade-off.
- **Mitigating Catastrophic Forgetting in LLMs with Forgetting-Aware Pruning (FAPM)** (2025)
  https://arxiv.org/pdf/2509.08255
  Prunes the "task vector" (Wft − Wpre) to remove redundant, forgetting-inducing directions; adapted to work for LoRA too (pruning the BA product). Strong empirical results (down to ~0.6% forgetting while keeping ~99.5% task accuracy).

## 7. Mitigation: Architecture-based / Mixture-of-Experts

- **MoE-CL: Self-Evolving LLMs via Continual Instruction Tuning** (2025)
  https://arxiv.org/pdf/2509.18133
  Dual-expert LoRA architecture: a per-task expert to preserve task-specific knowledge + a shared expert (with adversarial/GAN-based gating) for cross-task transfer.
- **Model Growth-based pretraining** — *Mitigating Catastrophic Forgetting in Continual Learning through Model Growth* (2025)
  https://arxiv.org/pdf/2509.01213
  Uses smaller-model "growth" to structure/expedite training of larger ones, evaluated for forgetting resistance — more of a pretraining-stage angle.

## 8. Scaling laws & mechanistic understanding

- **Scaling Laws for Forgetting When Fine-Tuning Large Language Models** (2024)
  https://arxiv.org/pdf/2401.05605
  Quantifies how forgetting scales with model size, fine-tuning data size, and update magnitude — a great empirical backbone if your project includes any predictive/quantitative modeling.
- **Understanding Catastrophic Forgetting in Language Models via Implicit Inference** (2023)
  https://arxiv.org/pdf/2309.10105
  Frames forgetting as a shift in the model's implicit task inference rather than pure knowledge loss — pairs well with the "spurious forgetting" idea below.
- Note on **"spurious forgetting"**: some recent work argues that apparent forgetting is often the model failing to recognize/invoke the old task format rather than erasure of the underlying knowledge — worth citing as a nuance/counterpoint in your intro.

## 9. Practical / applied summaries (non-peer-reviewed, useful for orientation)

- *Mitigating Catastrophic Forgetting in LLM Tuning* (apxml course notes) — clear walkthrough of rehearsal, EWC, and PEFT+rehearsal combos.
- *Avoiding Amnesia: Practical Guides to Mitigate Catastrophic Forgetting in LLMs Post-training* (Medium, Aug 2025) — practitioner-oriented summary covering spurious forgetting, rehearsal, and scale/architecture effects. Good as a "map of the landscape" but cite peer-reviewed sources for actual claims.

---

## Suggested reading order for a course project

1. Start with the two surveys (§3) to get the taxonomy.
2. Read McCloskey & Cohen + Kirkpatrick et al. for historical grounding (§1).
3. Read Luo et al. 2023 for the core LLM-specific empirical picture (§2).
4. Pick **one** mitigation family to go deep on based on your project's scope:
   - If you have compute for full fine-tuning experiments → regularization/replay (§4–5).
   - If you're doing LoRA/PEFT experiments (most tractable on limited compute) → §6, which is currently the most active sub-literature and has clear, reproducible baselines (O-LoRA, CURLoRA, OPLoRA, FAPM).
5. Use Scaling Laws for Forgetting (§8) if you want a quantitative/predictive framing for your evaluation.

## Possible project angles
- Reproduce and compare 2–3 LoRA-forgetting mitigation methods (e.g., standard LoRA vs. O-LoRA vs. FAPM-pruned LoRA) on a small model (1–3B) across a couple of downstream tasks, measuring forgetting on a held-out general benchmark (e.g., MMLU subset).
- Empirically test the "spurious forgetting" hypothesis: does restoring the original prompt format recover "forgotten" performance?
- Study the rank-vs-forgetting relationship from §6 (low-rank LoRA constraints vs. full fine-tuning) as a controlled ablation.
