# Approved damage-plus-diversity data selection

This job reuses the original cached teacher responses,existing KL scores and full-text BGE-M3 vectors. It does not load encoder/LLM weights,generate new responses,train or evaluate benchmarks. A cached tokenizer is used only to verify selected prompt tokens and display preserved responses.

Approved5November2026:common13,743 candidates;BGE-M3 revision5617a9f61b028005a4858fdac845db406aefb181;cosine farthest-first seed42;50:50 normalized damage/diversity ranks;128 whole prompts;no new length/quality filter;also export matching diversity-only128. The128-prompt inventory does not imply128 response tokens. Selection coefficient0.5 is separate from proposed later KL coefficient0.01.

## One NTU job

From the NTU terminal after uploading the archive:

```bash
cd /home/msai/hemanth005
tar -xzf AI6130_DamageDiverse_Selection_20261105_001.tar.gz
cd AI6130_DamageDiverse_Selection_20261105_001
mkdir -p logs
sbatch run_selection.sbatch
```

The job requests oneGPU,fourCPUs,12G host RAM and onehour. It uses the existing pinned environment at `/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python`. Direct execution avoids the earlier problematic nested job-step pattern. No package install,authentication or new model download is needed. Source/vector mismatch or missing cache stops the job for inspection;it does not regenerate data or silently change settings.

Check with the returned job number:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
tail -n 30 logs/damage-diverse-select-JOB_ID.out
tail -n 20 logs/damage-diverse-select-JOB_ID.err
cat results/damage-diverse-bge-rankfusion-20261105-001/status.json
```

Actual completion requires SlurmCOMPLETED/0:0 and application`state:complete`. Application`paused` is an intentional safe stop near55minutes or on a graceful signal. After that job ends,repeat the same submission;`sbatch run_selection.sbatch --resume` is an explicit alias. Completed stages are immutable and reused;numeric ranking state is saved every128 selections and on a requested pause. Do not submit concurrent copies into the same output folder. Laptop/VPN disconnection after submission does not manage the server job.

## Export locations

Under `results/damage-diverse-bge-rankfusion-20261105-001/`:

- `diversity-ranking/ranks.jsonl`:complete common-pool diversity ranking.
- `exports/damage_diverse_anchors.jsonl` and `damage_diverse_teacher_cache.jsonl`:combined128 and full original teacher token IDs/text.
- `exports/diversity_only_anchors.jsonl` and `diversity_only_teacher_cache.jsonl`:matching pure-diversity128.
- `exports/selection_manifests.json`:ranks,weights and whole-response inventories.
- `exports/summary.json`:actual lengths/EOS/token totals,KL ranges and A/B/set overlaps.
- `identity.json`,`exports/source_receipt.json`,stage receipts and runtime files:exact input/model/config/source/vector provenance and checksums.

The script verifies the original13,769 input/13,743 cache/26 skipped partition,exact audited teacher/cache/score bytes and step600 source,then the pinned BGE identity,full-text source/mapping,vector-stage hashes,normalization and passed8192-token length audit. Only selected candidate rows enter ranking. Existing evaluation/reference text is used by the old corpus verifier only for frozen provenance;no labels,answers,accuracy or forgetting outcomes enter selection.

Ranking uses FP64 with TF32 disabled. It implements the same cosine farthest-first initialization and1e-12 ID tie rule as the existing code;CPU synthetic tests verify numeric agreement and resume equivalence. The actual allocatedGPU and any resumed segment runtimes are recorded. Small hardware/software differences can affect near-tied numeric decisions;do not claim cross-hardware bitwise equivalence.

The vector cache is expected at `/home/msai/hemanth005/AI6130_Llama32_SemanticCalibration_20261027_001/results/llama32-semantic-calibration-20261027-001`;its preflight source is the existing AnchorPreflight folder. No cache mutation is permitted. Package integrity uses `files.sha256.json`;old source/cache/vector checksums are rechecked before completion. Final exports are atomically published and immutable.

## Boundaries and next stage

`training_ready:false` is intentional:this approved stage exports data only. It does not mean selection failed. Review the actual128 membership,teacher outputs and token totals before preparing the separate concrete training package. Proposed later controls remain fresh pinned3B/seed42,9619/original300,3609updates,lambda0.01,64positions every4updates/57728exposures and fixed five tests/GSM8K256;training is not authorized by this archive.

BGE diversity differs from the teammate's MiniLM reference. Use the matching BGE diversity-only control for a clean selector comparison,or disclose the encoder difference. Original300 overlap,one-seed limitations and imperfect teacher responses remain. Rank fusion is not MMR and does not guarantee that its final subset maximizes pairwise diversity. No test-driven weight/filter tuning,no factual-correctness certification and no proof of improved retention is claimed by selection alone. Preserve all A/B models/results.
