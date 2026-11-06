# Approved original300 damage-A coefficient calibration

Hemanth approved: **600 updates each, 1,800 total**, first disposable eight-update
hardware checks, then fresh task-only, lambda0.01 and lambda0.05 runs. This is an
exploratory calibration, not the final four-method retention comparison.

## Fixed recipe

| Setting | Value |
|---|---|
| Base | pretrained meta-llama/Llama-3.2-3B at 13afe5124825b4f3751f836b40dafda64c1ed062 |
| Initialization | Independently fresh base + new LoRA; seed42 in every condition |
| Data | Original friend9619 train /300 validation, byte-identical supplied files |
| LoRA | q/k/v/o; rank128,alpha256,dropout0; BF16,no quantization |
| Optimizer | AdamW5e-5; weight_decay0.01; betas0.9/0.999; eps1e-8; clip norm1 |
| Target batch |1 example ×8 accumulation; original per-epoch seed42+epoch shuffle |
| Scheduler | Original361-update warmup /3609-update linear schedule; stop at600 |
| Target format | Question/Answer, no extra BOS, reference answer +native EOS, limit2048 |
| Anchor set | Exact job51061 unrestricted top128;255 unique response positions |
| Anchor KL | Full-vocabulary teacher→student, T1, response-token mean, EOS included |
| Exposure |64 positions every4 optimizer updates;150 updates/9600 positions per anchored run |
| Validation | Original300 reference loss at updates0/300/600; no generated benchmark answers |

Anchor loss uses the same saved teacher prefixes, never independently generated
student prefixes. Teacher weights are frozen. Scores choose membership only;
they do not additionally weight training loss. Memory adaptation: one anchor
example per microbatch, accumulating sums divided by all64 scored positions.
Only scored rows are converted to FP32 probabilities. CPU value/gradient tests
verify equivalence with the existing full-vocabulary KL implementation.

## Preserve existing work

The archive does not patch the repository, original protocol, old models,
teacher caches, KL scores or selected set. All new outputs live under this
package's `results/damage-A-kl-calibration-20261104-001/`. The original step600
adapter identifies the source ranking; it is never loaded into the new student.
The completed clean229 reproduction remains separate and is not used here.

The eight-update smoke uses the longest training examples and longest selected
anchor prompts as disposable fixtures. It does not evaluate the research
validation set. Checkpoint state is round-tripped at update4 and then training
continues through update8. All three smoke students are discarded for research
initialization. Real hardware checks remain pending until submitted on NTU.

## First: transfer and smoke

Transfer `AI6130_DamageA_KLCalibration_20261104_001.tar.gz` from your local Mac
to your NTU home directory. In your NTU terminal:

```bash
cd /home/msai/hemanth005
tar -xzf AI6130_DamageA_KLCalibration_20261104_001.tar.gz
cd AI6130_DamageA_KLCalibration_20261104_001
mkdir -p logs
sbatch --time=01:00:00 run_calibration.sbatch smoke
```

After submission, use the actual job ID:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
tail -n 35 logs/kl-calibration-JOB_ID.out
tail -n 20 logs/kl-calibration-JOB_ID.err
```

The hardware-smoke summary must report `complete` and all three eight-update
conditions must pass. Do not mistake a successful partial Slurm job for full
application completion. Use the report command below to inspect all conditions.

## After the smoke is reviewed: calibration

From the same new package directory:

```bash
sbatch run_calibration.sbatch calibrate
```

The three conditions run sequentially in one allocation. No queued-job chain
or laptop connection is needed after submission. If time remains insufficient,
the job commits state, reports `paused`, and exits normally. Repeating the same
command resumes the first unfinished condition; completed conditions are reused.
`--resume` is also accepted by the Python CLI but is not required.

Each allocation has a six-hour Slurm limit and a5.5-hour internal calibration
budget. Smoke has a45-minute internal budget. Signals request a pause after
the current unit of work. Recovery records parameters, optimizer moments,
scheduler, RNG, exact target cursor, anchor cursor, exposure and validation phase.
A killed job may replay work after the last50-update committed checkpoint; no
uncommitted history is reported as completed. Validation resumes item by item.

Runtime identity requires matching GPU model and software across the smoke,
conditions and resumed jobs. Node names are recorded but need not be identical
if the GPU model matches. If NTU allocates a different GPU model, the job stops
for inspection instead of silently mixing A40 and L40S results. Submit on a node
with the recorded hardware; do not edit identity files to bypass the check.

## Report and curves

Lightweight status reporting, without importing model libraries:

```bash
/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python -B -S calibrate.py report
```

Each condition has committed `training_history.json`, `validation_history.json`,
per-item validation records, adapters at300/600, and independent recovery state.
No checkpoint is automatically selected across coefficients. The original
three-epoch scheduler is retained if a later continuation is approved; this
package stops at600 even when resubmitted.

Calibration creates `curves/loss_curves.png`, `.svg`, `total_objectives.png`,
`curves.csv` and `report.json`. Curves are regenerated after normal pauses and
completion. If a hard kill prevents plotting, resume first; the committed JSON
history remains authoritative. Explicit plotting should run on an allocated
node or a local machine with Matplotlib, not as heavy work on the head node.

The main figure separates math loss, original300 validation loss, raw training
anchor KL, fixed-selected-set KL, weighted KL and gradient norms. The fixed-set
KL is measured on the training anchors, including for the task-only control;
these diagnostic forwards do not contribute gradients or training-token exposure.

## Interpretation

- Lower total loss cannot select a coefficient because lambda changes the objective.
- Lower KL on the selected anchors is not retained benchmark capability.
- Original300 contains70 training-question matches and one additional internal
  duplicate. It is retained for the explicitly approved reproduction and is
  not described as independent unseen validation.
- These600-update checks can detect failure or severe suppression of math learning;
  they cannot establish final retention or disprove later benefits.
- No official test labels or outcomes enter training or coefficient selection.
  A separate retention-development protocol would require agreement first.
- One seed and a tiny, mostly short-response anchor set limit conclusions.
- Matching scored-token exposure does not guarantee equal wall time or unique
  selected content across the later four methods. Final study controls remain open.

The protocol, source hashes, actual software/GPU, source selection metadata,
initialization hashes and exact data-order fingerprint are saved. No coefficient
winner or improved accuracy is claimed by preparing or running this package.
