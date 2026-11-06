# Approved fixed-model evaluation:friend-style256 tokens,five datasets

Hemanth approved this protocol on5November2026. No new training,checkpoint
selection,anchor selection or coefficient tuning occurs here. Earlier models,
datasets,reported results and source packages are read-only.

## Models and source checks

1. Untouched `meta-llama/Llama-3.2-3B`,revision
   `13afe5124825b4f3751f836b40dafda64c1ed062`.
2. Original teammate task-only step2400,weightsSHA256
   `e6196178a27f3ed4e86c5bc2259fe1f6cd19a6d6c5ef4510595c7287496caf61`.
   Use the preserved NTU checkpoint;fall back to the exact reviewed Hub upload
   revision only if that local directory is absent. Wrong local bytes fail;
   they are not silently substituted. This is not the fresh clean229 reproduction.
3. Our completed damage-A lambda0.01 step2400,selected by the original300
   reference-loss rule. The complete3609-update trajectory,all14 validation
   records,model selection and checkpoint-file receipts are verified before
   loading it. No final test selected this checkpoint.

The chosen damage adapter saw38400 scored anchor positions at2400;the full
trajectory used57728,128 prompts/255 unique response positions,64/every4.
Keep this label. Other selector budgets have not been verified as matched;
this comparison cannot establish damage-selection superiority over different-
budget random/diversity runs. Any future harmonization must be agreed from
methodological controls,not chosen from these final-test outcomes.

## Frozen evaluation

| Benchmark | Split | Questions per model |
|---|---|---:|
|GSM8K|test|1319|
|ARC-Easy|test|2376|
|ARC-Challenge|test|1172|
|OpenBookQA|test|500|
|SocialIQa|labelled validation|1954|

Each model answers7321 questions;21963 predictions across all15 cells.
SocialIQa's labelled validation split is used as a retention evaluation set;
it is separate from MATH-10K's original300 training-validation set.

GSM8K:zero-shot `Question: {question}\nAnswer:`,no added BOS,greedy generation,
native EOS,maximum256 new tokens,original teammate final-number parser. Raw
responses,generated token IDs,EOS and cap diagnostics are saved. Cap hits do
not automatically mean wrong answers. Earlier development evidence favoured512
for two questions;256 is the explicitly chosen team-matching profile. Do not
change this cap after seeing final scores.

Multiple choice:exact reviewed common template:

```text
Question: {question}
Options:
A. {choice_A}
B. {choice_B}
...

Select the correct answer option.
Respond with only the option letter.
Answer:
```

Use the actual dataset's labels,including numeric ARC labels where present.
Score separately encoded leading-space labels by summed conditional log
likelihood,without length normalization,FP32 logsoftmax,first-choice ties.
These are probability-based choices,not freely generated-letter parsing or
full-answer likelihood. The snapshot of the teammate evaluator,backend and
numeric parser is unchanged and content-hashed. All three models share prompts,
scoring,full dataset rows,seed42,BF16,no quantization and the pinned tokenizer.
Old five-shot/512-token results are not reused under this different fingerprint.

## One allocation:checks,probes,then testing

Transfer/extract the separate archive in your NTU home. In the extracted folder:

```bash
mkdir -p logs
sbatch --nodelist=TC2N01 run_evaluation.sbatch
```

The existing pinned NTU environment is used. TC2N01 is a known matchingA40
node. The job refuses model computation on the head node. It first verifies
sources,prepares all fixed inputs and audits their lengths without truncation.
It records exact normalized input overlaps with actual math training,
original300 validation and selected anchors. Flags do not change membership
or automatically exclude anything;this is not semantic/pretraining certification.

It then runs synthetic MC/eight-token-generation probes for all three models.
After they pass,it automatically continues through real evaluations in the
same allocation:base's five benchmarks,task-only's five,damage-A's five.
No manual command between benchmarks or second smoke submission is needed.

The Slurm limit is6hours;the internal budget is5.5hours including checks/probes.
Signals request a pause after the current question. Repeating the same submission
resumes unfinished work. Each per-item prediction is flushed/fsynced with its
row/identity checksum. A torn uncommitted partial line can be removed;completed
outputs are rejected on corruption and are never silently repaired/overwritten.
Source/GPU/software changes stop reuse rather than mix incompatible results.
The per-cell ETA printed in logs applies to that benchmark/model cell only.

## Status and files

```bash
squeue -j JOB_ID
tail -n 20 logs/damage-A-eval-JOB_ID.out
tail -n 15 logs/damage-A-eval-JOB_ID.err
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
/home/msai/hemanth005/AI6130_Srihari_Step5_20261102/code_git_ab02b28b/.venv/bin/python -B -S evaluate_run.py report
```

All new output lives under
`results/damage-A-step2400-friend256-five-20261105-001/`:

- `identity.json`,`evaluation.json`:fixed model/protocol/hardware provenance.
- `prepared/<benchmark>/`:raw input rows,source revisions,context audit,overlap flags.
- `<mode>/<benchmark>/items.jsonl`:raw responses or option scores,explicit gold,
  correctness and per-item fingerprints;plus summary/completion receipts.
- `status.json`:complete cells,remaining work and all reported accuracies.
- `summary/accuracy.png`,`.svg`,`.csv`:completed-cell accuracy graphs/tables;
  missing cells are not plotted as zero.
- Paired damage-versus-task-only gains/losses are reported when complete;
  preserve the predictions for later uncertainty analysis.
- `jobs/`,`failures/`,`preservation.json`:runtime/job evidence and unchanged sources.

`report` and`preflight` are lightweight and import no model libraries.
Full completion means`state:complete` with`remaining:[]`. SlurmCOMPLETED alone
can also describe a clean time-budget pause;inspect application status too.

## Limits and source/licence provenance

No accuracy or retention outcome is claimed by this preparation. One seed,the
original300's known overlap,mostly short training anchors,the external task-only
baseline's provenance and prior exploratory use of evaluation benchmarks must
be disclosed. Report every benchmark and math/retention trade-off;do not tune
lambda,anchors,checkpoint or cap on the resulting final failures. Matching
evaluation does not by itself fix differing anchored-training budgets.

Dataset versions/splits are pinned in `evaluation.json`;GSM8K,ARC andOpenBookQA
are loaded from their specified HF revisions. SocialIQa uses the same official
labelled development examples from the public AI2 archive,with archive,input and
label-file SHA256 checks,to avoid an obsolete dataset-script dependency. This
source/label mapping was previously validated in this project;all labels are
explicit gold used for scoring only,not given to the model.

Source repository and model/dataset licence pointers are retained in the
project's `data/README.md` and earlier source packages. The unchanged teammate
implementation is credited in `snapshot_provenance.json`. Meta Llama3.2 access
and original data licences still apply. The archive distributes no model
weights,downloaded benchmark datasets,teacher cache or credentials. Use the
existing authorized NTU model cache/account. No GitHub publication is performed.
