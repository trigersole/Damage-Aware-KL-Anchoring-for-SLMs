# Matched Random anchor training on NTU

The supplied seed-42 preview selects 128 whole responses from the common 13,743-entry teacher cache. Its 22,100 unique response positions include 61 continuations ending in native EOS. Another 67 reached the original 256-token cap. The selection order and exact token lengths were reproduced locally using the frozen ranked-pool export. NTU verifies the canonical cache bytes and reconstructs the ordered rows before loading models.

Training reuses the approved Damage + Diverse math and KL update, target data and order, fresh Llama-3.2-3B LoRA initialization, original 300-example validation, and 3,609-update schedule. Teacher-to-student KL uses coefficient 0.01 and 64 response positions every four updates. The full run scores 57,728 positions. Unique coverage and repetitions per position differ with the selected responses. The original validation overlap and one-seed limitation still apply.

## NTU sequence

Damage + Diverse evaluation job 51587 completed. Use the finalized Random package in a separate NTU directory. The execution approval binds this exact plan and the unchanged teammate selection manifest.

```bash
mkdir -p logs
python -B -S run_random.py preflight
sbatch --time=01:00:00 run_random.sbatch smoke
```

The smoke check uses the longest selected input, verifies recovery and ends after eight updates. After its completion record shows `step: 8`, `anchor_positions_seen: 128`, and `recovery_probe_passed: true`, submit full training.

```bash
sbatch run_random.sbatch train
```

Full training starts from a fresh base and uses the supplied seed-42 order. A pause can be resumed with `sbatch run_random.sbatch train --resume`. At 3,609 updates and 57,728 exposures, the original300 minimum-loss rule selects the checkpoint. Run the frozen five-benchmark evaluator after Slurm confirms a clean exit and the application reports `complete`.

```bash
sbatch run_random.sbatch evaluate
```

The evaluator reuses the audited base, task-only, and Damage A records and evaluates only the selected Random adapter. It uses the same GSM8K, ARC-Easy, ARC-Challenge, OpenBookQA, and Social IQa settings as Damage + Diverse. Resume a paused evaluator with `sbatch run_random.sbatch evaluate --resume`.

The launcher uses the matching NTU A40 environment, 30G host RAM, and a six-hour allocation with a safe pause budget. Slurm logs are under `logs/random-anchors-<job>.out` and `.err`. The training and evaluation use distinct Random result directories under `results/`.
