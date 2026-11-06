# Approved damage-A lambda0.01 full-training continuation

**Hemanth approved this full-budget continuation on4November2026. No new training
has been submitted by preparation.** The agreed coefficient is0.01. The approved
release includes exact-plan authorization, so no additional authorization command
or confirmation is needed. The original proposal archive remains preserved.

| Item | Approved next run |
|---|---|
| Scope | Damage-aware VariantA only; teammates handle other selectors |
| Parent | Our fresh lambda0.01 calibration from job51090, completed update600 |
| Duration | Continue to3609 total updates, three epochs;3009 additional updates |
| Math data | Exact original9619 MATH-10K rows; no resplit or new questions |
| Validation | Friend's original300 rows; same response-token-weighted reference loss |
| Initialization provenance | Parent began independently from pinned untouched3B with fresh LoRA |
| Continued state | Adapter, AdamW moments, scheduler, RNG, target/anchor cursors preserved |
| LR schedule | Original3609-update linear schedule/361 warmup; no reset or extension |
| Model | meta-llama/Llama-3.2-3B,13afe5124825b4f3751f836b40dafda64c1ed062 |
| LoRA | q/k/v/o,r128,alpha256,dropout0,BF16,no quantization,no gradient checkpointing |
| Optimizer | AdamW5e-5,weight_decay0.01,betas0.9/0.999,eps1e-8,clip norm1 |
| Batch/order | One example ×8 accumulation; unchanged exact per-epoch shuffles,seed42 |
| Prompt/sequence | Original Question/Answer,no extra BOS,reference+native EOS,max2048 |
| Anchors | Unchanged job51061 top128 and saved teacher prefixes;255 unique response positions |
| KL | Teacher→student,T1,response-token mean,EOS included,lambda0.01 |
| Exposure |64 scored positions every4 math updates;57728 total,48128 additional |
| Checkpoints/validation | Every300 updates and final3609; inherit0/300/600 records unchanged |
| Model selection | Lowest original300 reference loss among trained checkpoints; exact tie chooses earlier step |
| Benchmark tests | Separate later stage; no generated GSM8K/retention answers here |

The full data budget is28857 math question presentations, with24057 remaining.
Partial epoch-end batches contain3 questions, so it is not3609×8 presentations.
The selected255 unique anchor positions are cycled repeatedly;57728 is scored
training exposure, not a count of unique prompts or prompt-input tokens.

## Preservation and handoff

This runner imports verified functions from the unchanged calibration package.
It checks every pinned parent source/data hash, the original Step5 source/cache
guards, the exact reviewed lambda0.01 validation result and matching GPU/software.
It reads the completed `calibration/damage_001` recovery state into a new run
folder, commits an independent copy and compares parameters, optimizer, scheduler,
RNG and progress before/after the round trip. Original calibration files are
never edited. New outputs are under this package's
`results/damage-A-lambda001-full3ep-20261104-001/`.

This is our KL-trained600-update prefix. It is not the friend's original
task-only step600 adapter used to discover vulnerable anchors. Lambda stays0.01
for every inherited and additional update. The prefix was independently fresh
from the untouched checkpoint, as required for comparisons; continuation simply
resumes that same trajectory. It does not add three more epochs after600.

All inherited records remain traceable through `origin.json` and the original
read-only checkpoint paths. New scientific adapters are immutable. Working
recovery generations in the new folder retain the latest two copies. `best_model.json`
points to the selected verified adapter, including a parent adapter if it wins.
No Hugging Face upload or existing `best_model` is modified.

## Approval and launch

The full3609-update budget and selection rule were explicitly confirmed in the
project chat. The approved archive includes `execution_approval.json` matching
this exact plan. `train` still refuses a missing or mismatched approval record;
do not edit the plan or approval to bypass that guard. This is the record of the
approval already given, not another user permission step.

Transfer/extract the `_APPROVED.tar.gz` release in your NTU home directory,
then optionally use lightweight preflight from its directory:

```bash
mkdir -p logs
/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python -B -S full_train.py preflight
```

Launch one direct job from the extracted folder:

```bash
sbatch run_full.sbatch
```

It must receive the same GPU model/software as the parent (reportedA40).
If a different GPU is allocated, the guard stops. Resubmit to a matching node;
do not alter identity files. A known parent node isTC2N01.

```bash
sbatch --nodelist=TC2N01 run_full.sbatch
```

No computation is launched on the head node. The Slurm script directly executes
Python without `srun`, following the successful calibration launcher. Each
allocation pauses near5.5hours within a6hour limit and commits exact recovery.
Repeat the same submission after a pause; completed runs exit without retraining.
Only one allocation is needed at a time; no dependency chain is queued.

## Monitoring and output

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
tail -n 20 logs/damage-A-full-JOB_ID.out
tail -n 20 logs/damage-A-full-JOB_ID.err
/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python -B -S full_train.py report
```

Successful import prints `[HANDOFF] verified; resume step=600` and training
continues at601. Recovery state is saved every50 updates and at validation
boundaries. A hard kill may replay work after the last commit. Validation
resumes per question; inherited metrics are not recomputed under new identities.
The GPU handoff is checked before its first additional optimizer update.

After a normal pause/completion, `curves/` contains PNG/SVG figures,CSV and a
compact report. Updates1–600 are explicitly marked as inherited. Model selection
uses reference loss only, never total loss or official-test outcomes. Retain
per-item validation records and immutable checkpoint receipts for later audits.

## Scientific limits

- This remains the explicitly chosen original300 branch:70 questions overlap
  training prompts and an additional internal duplicate is known. Do not claim
  unseen validation independence.
- Lower training-anchor KL does not establish better retention or factual teacher
  correctness. The128 selected teacher continuations are mostly one/two tokens.
- One seed, a shared coefficient and a token-matched cadence do not establish
  the full learning/forgetting trade-off. Later evaluation must report all
  benchmarks under a frozen common protocol and compare math performance.
- For selector comparisons, teammates must match the final scored-token exposure,
  data, LoRA, optimizer, schedule and checkpoint-selection rule.128 prompts alone
  does not provide this match.
- Original benchmark tests are not rerun/tuned by this training package. The
  final suite and any additions remain separate decisions.
- CPU tests verify logic/equivalence on synthetic fixtures, not real full training.
  No full CUDA continuation or completed3609-update result is claimed here.
- Exact state round-trip checks do not guarantee bitwise agreement of future
  CUDA operations; nondeterministic kernels/rounding can still affect results.

## Dependencies and attribution

Use the existing pinned NTU Python environment and caches. The separate archive
contains no model weights, datasets, credentials or duplicated teacher cache.
`parent_files.sha256.json` credits and freezes the prior package, whose
`source_provenance.json` and `DATA_AND_LICENSES.md` describe teammate code,
MATH-10K source provenance, Dolly CC BY-SA3.0 and Meta Llama3.2 access/license.
No new source dataset or license interpretation is introduced by continuation.
