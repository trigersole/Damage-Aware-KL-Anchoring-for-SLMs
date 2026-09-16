# Team G4 proposal draft supplied by a teammate

> Status: Source snapshot supplied in the planning chat. Preserve this document as the team's concise draft; use `PROJECT_CONTEXT.md` and `DECISIONS.md` to track later clarifications.

## AI6130 Large Language Models – Course Project Proposal

**Title:** Can Vulnerability Guided KL Anchors Reduce Catastrophic Forgetting in Parameter Efficient Fine Tuning?

**Team members:**

- Sridhar Srihari (G2510717L)
- Dadi Hemanth (G2510794D)
- Siddharth Paliwal (G2510801L)
- Balasubramanian Hariharan (G2610112K)

## Goals/Objectives

Fine tuning of language models is generally done when we want to adapt the model to a specific downstream task. In doing so, it is generally observed that the model exhibits “catastrophic forgetting”, a phenomenon by which the model loses its previous knowledge or capabilities. PEFT methods like LoRA customize a pretrained model to the downstream task by training a fraction of the parameters while the rest are frozen. Although this is known to mitigate catastrophic forgetting to a certain extent, further improvements are still possible and research is widely conducted regarding the same.

Our proposed project aims to further mitigate catastrophic forgetting by the usage of vulnerability-guided KL anchors while fine-tuning models for a particular task. The key idea is that certain general-purpose capabilities might show greater behavioural changes than the rest. We plan to identify such capabilities by measuring the divergence between the original pretrained model and task-adapted model on a pool of general-purpose prompts.

The prompts which show maximum divergence between original and task-adapted models are then used as KL anchors for a fresh LoRA fine-tuning run, and we call these Damage KL anchors. We hypothesize that the resulting model will achieve comparable performance on the downstream task while retaining its generalization capabilities better than a purely task-specific LoRA fine-tuned model, thereby mitigating catastrophic forgetting.

In order to determine if the vulnerability-guided selection is beneficial, we plan to compare it with ordinary task-only LoRA fine-tuning, random KL anchor selection and diversity-based KL anchor selection. This helps us to assess if selecting anchors based on observed behavioural damage gives an advantage over selecting anchors without considering model vulnerability.

**Model:** Qwen2.5-1.5B

## Datasets

- GSM8K – target dataset for the fine-tuning task
- BoolQ, HellaSwag and ARC-Easy – datasets to estimate model-retention capabilities and measure catastrophic forgetting

## Baselines

- Base model and task-only LoRA fine-tuned model
- Random-KL-anchor and diversity-based-KL-anchor guided LoRA fine-tuned models

## Evaluation

Evaluated on target-task learning capabilities and retention of previously acquired capabilities.

## References

1. Hu et al., *LoRA*, ICLR 2022.
2. Biderman et al., *LoRA Learns Less and Forgets Less*, TMLR 2024.
3. Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015.
4. Li and Hoiem, *Learning without Forgetting*, ECCV 2016.
5. Jin and Ren, *Demystifying Language Model Forgetting with Low-rank Example Associations*, NeurIPS 2025.
6. Zheng et al., *Spurious Forgetting in Continual Learning of Language Models*, ICLR 2025.
