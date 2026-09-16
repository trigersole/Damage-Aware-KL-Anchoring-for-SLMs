# Focused prior-art check — 2026-09-16

This is a scoped check for the current experiment, not a proof of novelty. The contribution should be phrased as a **controlled investigation** of whether preliminary behavioural-drift ranking helps allocate a limited KL anchor-token budget during LoRA adaptation.

Relevant neighbours found in primary sources:

- [Mix-CD (Findings of NAACL 2025)](https://aclanthology.org/2025.findings-naacl.138/) prioritizes collateral-damage examples for rehearsal. It is related to damage-guided selection; the present study asks about unlabelled prompt selection for token-distribution KL anchoring under a fixed token budget.
- [Demystifying Language Model Forgetting with Low-rank Example Associations (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/06872e1e6d11baf2ae27285c50132f4f-Abstract-Conference.html) studies and predicts forgetting for replay. This makes a broad claim that “vulnerability-guided preservation” is new inappropriate.
- [Context-Free Synthetic Data for language model preservation](https://arxiv.org/abs/2505.13811) studies KL-based preservation on synthetic inputs.
- [ASFT](https://arxiv.org/abs/2509.23753) uses KL control of adaptation-induced drift.
- [Anchored Learning](https://arxiv.org/abs/2605.04468) uses dynamic distribution anchors.

The searched sources did not establish an exact match to this project's planned comparison of preliminary teacher-to-adapted KL ranking of unlabelled general prompts against random and semantic-diversity selection with equal scored anchor tokens during fresh LoRA runs. That is a search-limited observation. Recheck the literature before the final report, and report a null or negative result without changing the contribution framing.
