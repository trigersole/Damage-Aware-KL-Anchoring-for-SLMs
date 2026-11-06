# Matched Version B: smoke, fresh training, fixed testing

Approved by Hemanth on5November2026: “Approve matched B protocol (recommended).” No job is submitted by this archive. Previous code/models/results are read-only references. This separate B package must remain at its extracted NTU directory. The user submits each stage after the preceding stage completes; no extra queued jobs are required.

## Protocol

Pinned pretrained Llama-3.2-3B plus fresh q/k/v/o LoRA, r128/alpha256/dropout0, BF16, seed42. Original9,619 math training/original300 validation rows; exact per-epoch shuffling, batch1×8 with three-row final partial batches. Three epochs/3,609 updates/28,857 question presentations, AdamW5e-5, linear3,609 schedule,361warmup, weightdecay0.01, betas0.9/0.999, eps1e-8, clip1, sequence2048, no gradient checkpointing. Target response+EOS token-weighted loss; same A implementation.

Audited B128 anchors/1,749 unique response positions, >8-position cached response cutoff counting EOS; unchanged original KL ranking. Teacher→student T1 full-vocabulary KL, response/EOS mask, token mean, lambda0.01. Cycle the ranked flattened response positions:64 per event every4 updates,902 events/57,728 total exposure. No teacher generation/rescoring or additional quality filtering.

Evaluate original300 reference loss at0,300,…,3600,3609; select the trained checkpoint with minimum loss, ties earlier. The fixed1,749-position training-anchor KL is only a training diagnostic, not independent retention performance. Known validation overlap is explicitly disclosed.

## Prerequisites

Use NTU VPN/SSH and the existing pinned Step5 environment and authorized model cache. All model work runs inside Slurm. The script requests TC2N01 and requires NVIDIA A40/runtime matching A. Do not edit identity/configuration files to bypass a failure. Existing A calibration, completed B selection and A evaluation folders must remain at their recorded paths in `plan.json`; all are verified read-only.

## Submit the disposable hardware check first

From your LOCAL MAC terminal, upload the adjacent archive to `/home/msai/hemanth005/` using scp. Then run these in the NTU terminal:

```bash
cd /home/msai/hemanth005
tar -xzf AI6130_DamageB_Training_20261105_001.tar.gz
cd AI6130_DamageB_Training_20261105_001
mkdir -p logs
sbatch --time=01:00:00 run_b.sbatch smoke
```

The smoke performs8 updates with64 training-only math presentations/128 scored anchor positions. It stresses the longest training sequences and the selected longest total anchor sequence/longest teacher response. At update4 it saves/reloads parameters, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG and cursors; then performs following updates. It computes no validation/test metric. CPU tests are separate and cannot certify GPU memory.

Check with the actual returned job number:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
tail -n 20 logs/damage-B-JOB_ID.out
tail -n 15 logs/damage-B-JOB_ID.err
cat results/damage-B-gt8-lambda001-full3ep-20261105-001/hardware-smoke/status.json
```

Success requires SlurmCOMPLETED/0:0, application `status:complete`, step8, examples64, anchor positions128 and `recovery_probe_passed:true`. This receipt does not mean full training is complete.

## After the smoke passes: fresh full B training

Already authorized; no new recipe decision is needed. Submit only after the smoke job has ended:

```bash
sbatch run_b.sbatch train
```

Full B independently reloads untouched weights and reinitializes LoRA at step0. The smoke state is quarantined as a disposable fixture and is never loaded into full training. A/discovery adapters are never used as initialization. Both initial LoRA parameters and target order are checked against A.

The wrapper directly executes Python, without srun. It pauses after about5.5h or USR1, within a6h allocation. At a safe update boundary it commits model/optimizer/scheduler/RNG/target and anchor cursors; validation saves every item. It saves periodic recovery every50 updates; a forced kill can lose up to49 in-memory updates, but repeating uses the last committed state. For a graceful pause, `scancel --signal=USR1 --batch JOB_ID`, then wait for the allocation to end.

If application status is `paused`, resume with the same stage:

```bash
sbatch run_b.sbatch train --resume
```

`--resume` is an explicit alias; repeating the stage already resumes safely. Do not submit duplicate jobs. Once complete, repeating is a verified no-op, not an extra epoch. Check `full-training/status.json` and `full-training/best_model.json` under the experiment output. Training/validation graphs and CSVs are in `full-training/curves/`.

## After full training completes: fixed selected-B testing

```bash
sbatch run_b.sbatch evaluate
```

Uses the same five frozen tests as A: GSM8K1,319, ARC-Easy2,376, ARC-Challenge1,172, OpenBookQA500, Social IQa labelled validation1,954. GSM8K: zero-shot `Question: {question}\nAnswer:`, no added BOS, greedy256/native EOS and unchanged final-number parser. MC: exact Question/Options/Select/Answer template; separate leading-space option-label encoding, summed likelihood, no length normalization, FP32 logsoftmax and first ties. No option-text scoring or free-generation substitution.

The85 audited A comparison files are pinned and verified. The evaluator audits all15 reference cells and reuses base/task-only/A results without inference or modification, then runs7,321 B predictions only. It verifies row identities for paired comparisons, pins the B receipt/selected step and saves per-item raw scores/token diagnostics. It processes as many benchmarks as fit in5.5h. Resume any remainder:

```bash
sbatch run_b.sbatch evaluate --resume
```

Output: `results/damage-B-selected-friend256-five-20261105-001/`. Accuracy PNG/SVG/CSV is in `summary/`. No training, checkpoint selection or test-cap tuning occurs during inference. Frozen256 cap remains a limitation; previous development512 diagnostics are not silently applied to this team profile.

## Lightweight report commands

```bash
DAMAGE_B_PYTHON=/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python
"$DAMAGE_B_PYTHON" -B -S run_b.py report
"$DAMAGE_B_PYTHON" -B -S evaluate_b.py report
```

These only read files. Do not run smoke/train/evaluation Python directly on the head node. You can disconnect VPN/close your laptop after Slurm submission; the server allocation continues independently. Completion of a stage does not submit the next stage automatically.

See `plan.json`, `protocol.json`, `execution_approval.json`, `reference_evaluation.json`, `snapshot_provenance.json`, `files.sha256.json`, and `DATA_AND_LICENSES.md` for the exact approved recipe, source identities, datasets and limits. No B GPU check/training/benchmark result is claimed by package preparation.
