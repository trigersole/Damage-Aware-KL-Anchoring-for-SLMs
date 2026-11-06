# NTU MATH-10K KL anchoring

These are source snapshots of the NTU packages used for the Llama-3.2-3B
MATH-10K study. The launchers, Python source and checksummed protocol files
retain the original package names and bytes so the team can audit what ran.
The Git baseline is `ab02b28be9f491c1cd8fb4cc025a91b26455a779`.

| Stage | Package | Slurm entry point |
| --- | --- | --- |
| Damage A selection | [AI6130_DamageA_Top128_20261104_001](AI6130_DamageA_Top128_20261104_001/) | `run_export.sbatch` |
| Damage B selection | [AI6130_DamageB_Top128_GT8_20261105_001](AI6130_DamageB_Top128_GT8_20261105_001/) | `run_export.sbatch` |
| Damage + Diverse selection | [AI6130_DamageDiverse_Selection_20261105_001](AI6130_DamageDiverse_Selection_20261105_001/) | `run_selection.sbatch` |
| Damage A calibration | [AI6130_DamageA_KLCalibration_20261104_001](AI6130_DamageA_KLCalibration_20261104_001/) | `run_calibration.sbatch` |
| Damage A continuation | [AI6130_DamageA_FullTrain_20261104_001](AI6130_DamageA_FullTrain_20261104_001/) | `run_full.sbatch` |
| Damage A evaluation | [AI6130_DamageA_Evaluation_20261105_001](AI6130_DamageA_Evaluation_20261105_001/) | `run_evaluation.sbatch` |
| Damage B | [AI6130_DamageB_Training_20261105_001](AI6130_DamageB_Training_20261105_001/) | `run_b.sbatch` |
| Damage + Diverse | [AI6130_DamageDiverse_Training_20261105_002](AI6130_DamageDiverse_Training_20261105_002/) | `run_dd.sbatch` |
| Random | [AI6130_RandomAnchor_Training_20261106_001](AI6130_RandomAnchor_Training_20261106_001/) | `run_random.sbatch` |

## Reproduce on NTU

Use the package README for the stage order and resume commands. The snapshots
reference the original NTU project paths and the pinned Step5 Python
environment. Place the matching prerequisites on NTU before submitting.
These include the original MATH-10K splits, the Step5 teacher cache, the
applicable anchor exports or ranking receipts, and any prerequisite
checkpoints and evaluation outputs. The package guard will reject missing
or mismatched inputs. The source paths and checksums are in the protocols
and file manifests. See the [data setup and licenses](../../data/README.md)
for the pinned source and exact split reconstruction.

For packages with `inputs/`, restore the two files at the paths recorded in
`files.sha256.json`. The manifests intentionally retain their original
entries. Training data, teacher caches, model weights, and raw results are
external to this source handoff.

## Matched controls

Damage A, Damage B, Damage + Diverse, and Random share the same training
update. Each trajectory starts from pinned Llama-3.2-3B with fresh LoRA.
Damage A resumes its own initial 600 updates. The matched run uses seed 42,
9,619 math training rows, original 300-row reference-loss validation, and
3,609 optimizer updates. KL uses the frozen teacher, temperature 1,
coefficient 0.01, and 64 scored response positions every four updates for
57,728 full-trajectory exposures. The 128 selected prompts can contain
different counts of distinct response positions. The independent
diversity-only teammate run requires separate verification.

The original validation split has known train/validation prompt overlap.
The archived protocols preserve that split for reproduction. The common
evaluation uses five fixed benchmark splits and selects checkpoints using
math validation loss before final evaluation.
