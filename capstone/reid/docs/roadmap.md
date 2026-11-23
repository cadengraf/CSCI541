# Joint vehicle ReID roadmap

Open [milestones.html](../milestones.html) for the complete dashboard, all reference-paper tables, local scores and copy-only commands. Refresh it with `../.venv/bin/python benchmark.py milestones`; this reads existing evidence and starts no model runs.

## Direction and run ownership

- The intended deliverable is a **shared detector / instance-mask / dedicated ReID network**. The existing ViT-L / Co-DINO / RefineMask model is a starting point. Independent TransReID is an accuracy reference and optional training-only teacher.
- **The user launches all long training and benchmark runs. The assistant must not launch, resume or schedule them without a new explicit instruction.** Full-split evaluations remain preferred for headline results.
- Full S02/S05 CityFlow validation finished in the user-launched unified suite: original specialist IDF1 11.35%, mixed AdaSP 11.20%, MSINet 2.64%. This weak ground-camera pipeline baseline is separate from aerial retrieval; the older paused queue is historical state, not the completed suite.
- Published accuracy metrics are valid references when labeled as reported results. Baseline retraining and exact author gallery assignments are **optional**, not prerequisites. Disclose local gallery and training differences once. Use controlled local measurements for a specific architecture or speed claim.

| Milestone | Current evidence | Exit gate |
|---|---|---|
| 1. Define the claim and metrics | Retrieval, recovery and shared-compute efficiency are distinguished. Published tables are in the HTML. | Choose one primary improvement; state data exposure, evaluation protocol and timing scope. |
| 2. Establish the specialist reference | Four mixed runs complete. Full VRU / local VRAI and local MSINet measurements exist. | Retain a reference and close the baseline sweep; no need to retrain every published model. |
| 3. Improve the actual joint ReID path | VRAI AdaSP: 33.62% held-out mAP versus matched triplet 27.66%; KITTI HOTA 78.42, sMOTSA 77.27. | Appearance improved; retain the HOTA tradeoff and recovery failures. Compare shared features against the specialist reference. |
| 4. Complete full aerial evaluation | Clean specialist results exist; intended joint full VRU, UAV-VeID and weather results remain open. | Full held-out joint scores against reported references plus a fixed local control. Extra-data and weather regimes are labeled. |
| 5. Prove correct video recovery | Small diagnostic: 2/3 forced returns versus 0/3 local reference; 12 correlated scored actions are insufficient. | More original-ID recoveries at comparable false-reconnection risk, with sufficient independent coverage. |
| 6. Persist target identity across restart / views | Memory machinery exists; rigorous restart/new-video evaluation remains open. | Reload trusted evidence, preserve the original ID or abstain, and prevent uncertain writes. |
| 7. Complete full tracking / cross-camera checks | Full suite CityFlow validation completed; best observed specialist IDF1 11.35%. | Diagnose poor association/rejection separately from aerial appearance training. No leading-method claim. |
| 8. Establish accuracy / cost tradeoff | Specialist is slower than MSINet for batched encoding; compact joint embedding offers potential memory savings. | Meet a declared accuracy budget and measure relevant cost reduction on the same hardware/workload. |
| 9. Assemble the paper evidence | Prototype, data and diagnostics exist; competitive joint results and ablations remain open. | Clear supported claim, controlled ablations, full evaluations, cost scope, provenance, runnable configs and limitations. |

## Active sequence

The matched VRAI metric test is complete: AdaSP improved held-out joint mAP to **33.62%**, versus **27.66%** for triplet continuation and **24.67%** for the earlier checkpoint. Forced recovery remains 2/3, with two false global claims per replay. Keep AdaSP as the joint retrieval reference; global recovery is not qualified.

1. **Next user run:** `python train.py rfdetr`, then `python benchmark.py rfdetr`. The pretrained RF front end reached HOTA 70.23 versus Co-DINO 77.98, but fresh video latency was 36.77 versus 127.46 ms/frame. Adapt its boxes/masks on training sequences with internal development selection. [Fine-tuning guide](rfdetr_finetune.md).
2. Inspect mask/tracking quality and both pipeline and fresh annotated-video latency. Label different training exposure/resolution/runtime; this is a practical replacement test.
3. If the front end is useful, prepare matched vehicle-mask fine-tuning where needed, then test a shared RF-DETR ReID adapter against the fixed specialist.
4. Test Cutie/McByte++ separately afterward. Do not repeat the completed adapter, identity-only or metric-loss pilots without a new reason.

`python benchmark.py suite` retains the original consolidated evaluation workflow; its completed run remains immutable. Scorecard updates are improvement-only. The user launches all long runs.

## Bounded follow-up if justified

Do not start another broad training sweep. Existing pooling, adapter, crop and distillation experiments already supply evidence. If an unresolved failure justifies one further joint ablation, match identity data, crop policy, scoring and update budgets, and decide whether to keep it before adding another change. Preserve immutable references alongside gated recent observations.

Use the specialist as a reference/teacher. A teacher contributes no inference cost only when absent online. EdgeTAM, if later enabled for mask propagation, counts as an online component; propagated masks do not establish persistent identity.

## Interpretation limits

- Current mixed TransReID+AdaSP clean VRU-Large: 75.14% mAP / 65.81% Rank-1; reported AdaSP ResNet-50+IBN: 95.9% / 93.3%. These are different recipes.
- VRU Small/Medium/Large refer to identity sets, not pixel sizes. Development scores are not the full Large benchmark.
- Historical compact crop-only joint scores are not scores of the shared ViT-L model.
- The missed nighttime target has about .40 original-reference cosine and .54 best earlier single-template cosine. This observed test does not justify lowering thresholds to force a match.
- Source-fitted thresholds yield roughly 8–10% unknown false acceptance on the small VisDrone transfer protocol. Retrieval alone does not establish safe reconnection.
- A paper need not win every table cell. It needs evidence for its own specific contribution; competitive joint ReID and an end-to-end speed improvement remain unproven.
