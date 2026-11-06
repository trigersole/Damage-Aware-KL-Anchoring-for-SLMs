# Approved matched damage-plus-diversity experiment

Approved5November2026:disposable8-update A40 hardware/recovery check,then independently fresh combined training and frozen five-benchmark selected-model testing. No job is submitted automatically. This folder and all outputs are separate from A,B,the original friend model and the completed selection package.

## Fixed recipe

Pinned pretrained`meta-llama/Llama-3.2-3B` revision`13afe5124825b4f3751f836b40dafda64c1ed062`,seed42,fresh q/k/v/o LoRA r128/alpha256/dropout0,BF16/no quantization. Original9619 MATH-10K training/original300 validation;three exact epochs/3609updates/28857 target presentations,batch1/accumulation8. AdamW5e-5,weight decay0.01,betas0.9/0.999,epsilon1e-8,clip1,linear3609 schedule/361warmup,2048 total-token limit,gradient checkpointing off,matching A/B.

Use the audited128 combined anchors/1453 unique response positions from selection51426,with original teacher token sequences intact. The50:50 rank weight is the already-completed selection rule;it is not the KL coefficient. During training use lambda0.01,teacher→student full-vocabulary KL,T1,mean over64 aligned response positions everyfour optimizer updates,including nativeEOS.902events/57728 total repeated exposures. Cycle through all1453 response positions;full-run counts differ by at most one across positions. Unique inventory and repeated exposure are reported separately.

All128 are kept as approved,including6EOS-only responses,49bare labels andthe`Tesrt` input. Other off-topic/factually unverified outputs remain. KL is a soft distributional constraint,not training on these strings as verified factual gold. No additional length/quality filter,teacher regeneration,KL-score recomputation or final-test tuning occurs.

Original300 validation loss is response-token weighted withEOS;evaluate at0,every300 andfinal3609. Select the minimum trained-checkpoint loss;exact tie earlier step. No test results select the checkpoint. The known train/validation overlap remains disclosed.

## Transfer and smoke first

From your local Mac terminal with NTU network/VPN,copy the adjacent archive to`hemanth005@10.96.189.12:/home/msai/hemanth005/`. Then in the NTU terminal:

```bash
cd /home/msai/hemanth005
tar -xzf AI6130_DamageDiverse_Training_20261105_002.tar.gz
cd AI6130_DamageDiverse_Training_20261105_002
mkdir -p logs results
sbatch --time=01:00:00 run_dd.sbatch smoke
```

Smoke uses training-only math fixtures and a reordered selected-anchor fixture including EOS-only,longest-total and longest-response examples. It tests8 updates/64 target presentations/128 scored anchor positions and a real-device model/optimizer/scheduler/RNG/cursor recovery round trip. No validation accuracy/test score is computed. All smoke state remains disposable;full training never initializes from it.

The script requests30G host RAM within current QoS,oneGPU/fourCPUs andknown A40 nodeTC2N01. It directly executes the existing pinned Python environment without nested`srun`. Nothing computes on the login/head node. A source,selection,model/software/hardware or approval mismatch stops execution rather than changing settings.

Check the returned job number:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
tail -n 25 logs/damage-diverse-JOB_ID.out
tail -n 20 logs/damage-diverse-JOB_ID.err
cat results/damage-diverse-lambda001-full3ep-20261105-001/hardware-smoke/status.json
```

Successful smoke requires SlurmCOMPLETED/0:0,applicationcomplete/step8/128anchor positions,and`recovery_probe_passed:true`. A matching smoke receipt gates full training.

## Fresh full run after smoke passes

```bash
sbatch run_dd.sbatch train
```

Full training reloads the untouched base and creates a fresh LoRA;it does not load smoke,A,B or discovery adapter/optimizer states. Initial trainable parameter hash and target data order must match the A/B reference. Only this exact full-run identity's own recovery can resume it. Model,optimizer,scheduler,RNG,counters and cursors are saved at validations,every50updates andsafe pause.

Each full allocation has6h hard limit and5.5h safe budget. If application status ispaused after the job ends:

```bash
sbatch run_dd.sbatch train --resume
```

Repeating the ordinary stage also resumes automatically;`--resume` is explicit shorthand. Do not submit concurrent copies. A paused application may still have SlurmCOMPLETED/0:0;Slurm reports process exit,while application reports remaining work.

Full results:`results/damage-diverse-lambda001-full3ep-20261105-001/full-training/`. Its source/protocol/initialization/data order,recovery,checkpoint receipts,loss curves and protected-file checks are preserved. Full-run`recovery_probe_passed:false` is the unchanged smoke-only probe flag;the separately successful smoke receipt gates the run. It does not mean a recovery test failed. The full-run progress counters and actual recoveries remain recorded.

## Fixed selected-model testing

After full training completes:

```bash
sbatch run_dd.sbatch evaluate
```

Test the frozen minimum-development-loss combined checkpoint on GSM8K1319,ARC-Easy2376,ARC-Challenge1172,OpenBookQA500 andlabelled SocialIQa validation1954. Use exact A/B friend profile:zero-shot`Question: {question}\nAnswer:`,no addedBOS,greedy256/nativeEOS on GSM8K;the reviewed Question/Options/select-letter/Answer prompt and summed leading-space label likelihood on MC tasks. Same pinned dataset versions,order,parser/scoring/runtime controls. Reuse the already-audited immutable base/task-only/A records;do not rerun or overwrite them. The completed B results remain separate for comparison;raw B per-item backup remains a separate pending audit.

Evaluation pauses safely after5.5h and saves each prediction. Resume with`sbatch run_dd.sbatch evaluate --resume` after the prior job ends. Evaluation output:`results/damage-diverse-selected-friend256-five-20261105-001/status.json`. Completion requires`state:complete`/emptyremaining,as well as successful Slurm exit. No training is performed by evaluate.

## Scientific and implementation boundaries

The math/KL/optimizer/schedule/recovery/evaluation kernels and target data are reused unchanged from A/B;new guards validate the different selected set and named experiment. Source provenance and local numerical/recovery/evaluation tests are recorded beside the archive. Actual GPU smoke/training/testing are not claimed by local tests.

Selection source must remain at`/home/msai/hemanth005/AI6130_DamageDiverse_Selection_20261105_001/results/damage-diverse-bge-rankfusion-20261105-001`;original Step5,A-calibration and A-evaluation sources are read-only dependencies. The script validates the published selection identities/receipts/full ranking,independently recomputes50:50 membership from original scores,and compares all128 cached responses to the canonical teacher cache before model work.

BGE diversity differs from teammateMiniLM;use the matching BGE diversity-only control or disclose the encoder difference before making selector-superiority claims. This package trains only the combined condition. Preserve the generated pure-diversity128 for future work;it is not automatically trained here. One seed,known original300 overlap,imperfect teacher prefixes and limited retention benchmarks constrain claims. Behavioural vulnerability is distribution drift,not evidence of damaged parameters or stored knowledge. No model weights,datasets or generated results should be committed to Git;this private reproduction bundle contains the unchanged approved target split and source/configuration,not weights or credentials.
