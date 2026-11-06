# Data,attribution and limitations

This private reproduction archive contains the already authorized MATH-10K training/original300 split and source/configuration snapshots. Do not commit datasets,the archive,weights,cached outputs or credentials. The original source is AGI-Edgerunners/LLM-Adapters,ft-training_set/math_10k.json,SHA2560342d0d860ad8592b579329337c90e42eefd3d9f2898043140cbd120630418b8. Preserve upstream row provenance and licences;repository licensing is not a blanket licence for all upstream content.

Anchors originate from databricks/databricks-dolly-15k revisionbdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a,CC-BY-SA-3.0. The selection remains on NTU and is verified by published receipts,stable IDs and full original teacher tokens. It uses the prespecified50:50 damage/diversity rank fusion over13743 candidates,128whole prompts/1453positions,no new length/quality filter. Dolly reference answers are not KL labels. Cached teacher outputs include empty/unmapped/off-topic and factually unverified content;they are not endorsed gold answers.

Model:https://huggingface.co/meta-llama/Llama-3.2-3B,pinned revision inprotocol. Meta Llama3.2 conditions apply. No weights,tokenizer files or credentials are bundled;existing authorized NTU caches are used. BGE vectors are reused from the completed selection;no encoder model is loaded by training.

Unchanged A/B computational helpers and the reviewed evaluator are attributed in snapshot_provenance.json. New guards bind the combined selection;no old A/B launcher is called. Fixed dataset versions/splits/counts and prompts/scoring are in reference_evaluation.json. Social IQa uses official labelled validation for evaluation. No final-test labels/results select anchors,tune weights or choose checkpoints.

Original300 has known training overlap. Single seed,imperfect teacher prefixes,unique-coverage differences andBGE-versus-teammateMiniLM encoding constrain inference. Match repeated exposure separately from unique inventory. Behavioural drift is not evidence of damaged parameters or stored knowledge. Preserve A/B and require matched random/diversity controls before claiming selector superiority.
