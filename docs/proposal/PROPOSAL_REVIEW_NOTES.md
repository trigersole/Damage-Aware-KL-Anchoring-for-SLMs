# Proposal review notes

These notes preserve technical clarifications discussed after the teammate draft. They are recommendations for the final protocol and report; they do not silently amend the submitted proposal.

## 1. Vulnerability-discovery model

The draft already says divergence is measured between the original and a task-adapted model and that anchors are used in a fresh LoRA run. The final methodology should additionally specify that the diagnostic model is temporary, how long it is adapted, and that no final comparison run starts from that checkpoint.

**Reason:** This avoids giving one condition extra target training and makes the two-stage design reproducible.

## 2. Meaning of KL comparison

The draft refers to divergence without defining what is compared. The technical protocol should state that the frozen teacher and current student score the same response-token positions and that KL compares their next-token probability distributions.

**Reason:** KL is not merely a comparison of two generated strings, which may have different tokens and lengths.

## 3. Concrete metrics

Recommended pre-specified metrics are:

- GSM8K: final-answer exact match
- BoolQ: accuracy
- HellaSwag: the locked harness's multiple-choice accuracy or length-normalized accuracy
- ARC-Easy: the locked harness's multiple-choice accuracy or length-normalized accuracy
- Forgetting on benchmark `b`: `score(base, b) - score(adapted, b)`

**Reason:** The course instructions request task-appropriate metrics, and pre-specification reduces selective reporting.

## 4. Scope of claims

Prefer **“retaining performance on the selected non-target benchmarks”** over **“retaining generalization capabilities.”**

**Reason:** BoolQ, HellaSwag and ARC-Easy probe particular behaviours; they do not establish preservation of every general capability.

## 5. Fairness controls

Random, diversity and vulnerability-guided anchored conditions should receive the same anchor-token budget and matched training settings. Retention should also be considered at approximately matched GSM8K performance or as a target-learning–forgetting trade-off.

**Reason:** A method can appear to retain more merely because it receives more preservation tokens or learns less of the target task.

## 6. Contribution statement

Recommended wording:

> Under a fixed anchoring budget, this project investigates whether behavioural vulnerability measured through task adaptation is a more effective anchor-selection signal than random or semantic-diversity selection for retaining non-target performance during LoRA fine-tuning.

**Reason:** KL regularization, distillation and learning without forgetting are established ideas. The proposed contribution is the selection strategy and its controlled evaluation.
