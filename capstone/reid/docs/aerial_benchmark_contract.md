# Primary aerial ReID benchmark contract

Reference: [Benchmarking UAV-based Vehicle Re-Identification under Simulated Weather Conditions](https://arxiv.org/html/2607.10583v1), [author code/data/checkpoint links](https://github.com/tranminhvu945/Benchmarking-ReID). This is the primary aerial retrieval reference. CityFlowV2 remains a complementary ground-camera tracking evaluation; its IDF1 cannot substitute for these retrieval results.

**User direction:** compare against published metrics without requiring baseline retraining or the authors' exact gallery assignment. Reproduction is optional. Disclose local gallery/training differences and call these reported-reference comparisons. Controlled local measurements are needed for a specific architecture/speed claim, and can use existing checkpoints. The user launches all long runs; the assistant must not launch, resume or schedule training/benchmarks without a new explicit instruction. The intended deliverable remains a shared detector/mask/ReID model; the separate specialist is a reference or optional training-only teacher.

## Verification completed

- Local VRU query+gallery image/identity pairs exactly match all **91,595** pairs in `test_list_8000.txt`: 83,595 query crops and 8,000 one-image-per-identity gallery crops.
- VRU Small, Medium and Large are **identity/gallery splits** (1,200 / 2,400 / 8,000 identities), not pixel-resolution categories. Source lists for all three are already available.
- Author MSINet logs report the same Large query/gallery counts. Its loader expects prepared `query_big.txt` and `gallery_big.txt`; neither file is currently present locally. Our gallery uses a fixed SHA256 choice with seed 42. Exact paper-gallery equality remains unverified.
- Local clean-VRU MSINet gives 89.93% Rank-1 / 93.77% mAP, close to the reported 90.0% / 93.8%. This supports protocol plausibility but does not prove exact gallery reproduction.
- The paper's AdaSP configuration uses **ResNet-50 with IBN**, identity classification and AdaSP loss. Our `mixed_adasp` is **TransReID ViT-B/JPM with an AdaSP metric term**. It is not a reproduction of their full model/training recipe.
- Authors' prepared weather datasets, exact prepared query/gallery lists, UAV-VeID data and the remaining checkpoint assets must be verified before reporting paper-matched results. Public links exist; availability of exact files is not yet confirmed by download.

## Accuracy targets from the reference tables

The strongest reported rows use AdaSP with ResNet-50+IBN. All values are percentages.

| Dataset/split | Condition | mAP | Rank-1 | Rank-5 |
|---|---|---:|---:|---:|
| VRU Large | Clean | 95.9 | 93.3 | — |
| VRU Large | Fog | 93.0 | 89.2 | — |
| VRU Large | Rain | 88.5 | 83.7 | — |
| UAV-VeID Test | Clean | 92.7 | 89.4 | 96.7 |
| UAV-VeID Test | Fog | 88.7 | 84.1 | 94.3 |
| UAV-VeID Test | Rain | 76.2 | 70.9 | 82.1 |

Do not compare our small development galleries to these values. Main evaluations use complete Large/Test splits; Small/Medium and UAV-VeID Validation provide supplementary/development evidence as appropriate. Gallery assignments, identity exclusions, transforms, normalization, checkpoint provenance, ranking distance, re-ranking and test-time augmentation must be explicit. Verify the shared scorer against the author scorer on the same extracted vectors, including one-positive AP and rank/tie behavior.

Current mixed TransReID+AdaSP on the local clean Large gallery gives **65.81% Rank-1 / 75.14% mAP**. It is not close to the reported strongest accuracy. Training conditions differ: our continuation uses VRAI plus an audited VRU subset with held-out development identities, whereas their recipe trains on the corresponding full weather-specific training split. Report this exposure difference and include a matched VRU-only control before making an architecture/data claim.

## Keep these training/testing regimes separate

1. **Paper-aligned matched conditions:** clean-trained→clean-tested; fog-trained→fog-tested; rain-trained→rain-tested, using the prepared weather protocol. These are the settings behind the cited table targets.
2. **Frozen clean-model corruption robustness:** one clean-trained checkpoint tested on clean/fog/rain. Useful, but not directly ranked against their separately weather-trained rows.
3. **Unseen-dataset transfer:** freeze a VRU-trained model and evaluate UAV-VeID without fine-tuning or target-threshold fitting. If UAV-VeID enters training, it no longer supplies unseen-dataset evidence for that checkpoint.

Extra training data is allowed in an explicitly labeled additional-data arm. Keep a matched-data control; do not silently merge extra images into every baseline and then attribute the result to architecture.

## Metrics to produce

- **Retrieval:** mAP, Rank-1, Rank-5 (Rank-10 supplementary), exact query/gallery counts, per-identity uncertainty or fixed-gallery seed sensitivity where relevant. Do not pick the best gallery seed. With one positive per identity, mAP equals mean reciprocal rank.
- **Weather:** absolute scores plus percentage-point drop from clean and the clean/fog/rain macro average, reported separately for the three regimes above. Use the same corruption assets for every method. A small drop from a weak clean model is not automatically better robustness.
- **Open-set identity:** correct known acceptance, wrong known acceptance, unknown false acceptance at frozen source-calibrated operating points, with identity-level uncertainty. This tests abstention, which closed-set Rank-1 alone cannot establish.
- **Video identity:** original-ID returns recovered/missed, false reconnections, confirmation delay, IDF1/HOTA where the dataset supports them. Keep these apart from crop-retrieval tables and synthetic weather measurements.

## Speed and memory comparisons

Re-run baselines on the same RTX 4090, input sizes, precision and batch sizes. The paper uses RTX 2080 Ti hardware; absolute milliseconds from different GPUs do not establish a speed win. Training-only heads/teachers are identified separately; any model running online counts toward latency.

Report batch-1 encoder median/p95 latency, batched throughput at declared batch sizes, GPU peak memory, loaded/invoked parameter counts, embedding dimension and gallery-search latency/storage. Separately report the full live pipeline's decode, detection, mask/propagation, ReID, search/association, rendering and checkpoint costs. Shared computation counts once. Cached-detection runs cannot establish detector-inclusive FPS.

Existing FP32 same-machine encoder measurements: original TransReID 4.43 ms at batch 1 / 105.96 ms at batch 32; MSINet 3.23 / 15.39 ms. The present specialist has not demonstrated an encoder-speed advantage. These are not full video FPS. A 3,840-D embedding also requires five times the raw gallery-vector storage of a 768-D embedding at equal dtype.

Pre-register an engineering tradeoff rather than claiming any faster but much less accurate model is superior. Example **proposed**, not achieved, gate: retain mAP and Rank-1 within one percentage point of the replicated reference while reducing relevant latency or memory by at least 2×. Alternatively, improve retrieval accuracy at comparable cost, or improve correct video recovery at the same false-reconnection budget. Choose the primary objective before tuning.

## Bounded execution order

1. Compare current full clean Large results against the published table, labeling local gallery and training exposure. No new baseline training or exact gallery download is a prerequisite.
2. Diagnose the actual shared ViT-L joint ReID branch using a matched local control and one targeted feature/training change. Preserve specialist and historical results as references.
3. The user runs complete weather splits and UAV-VeID Test once their assets and training regimes are fixed. A local weather protocol is valid when disclosed; claim exact paper reproduction only if its preparation actually matches.
4. The user runs targeted training/efficiency changes against a declared accuracy/cost gate, with checkpoint selection confined to development data. Keep video recovery and CityFlow as additional evidence. Existing baseline weights can be timed locally without reproducing their training.

The assistant-started CityFlow queue was stopped at the user's instruction, preserving committed checkpoints. The user decides when to resume it or start other long runs. Do not start competing GPU timing jobs; use tmux, checkpoints and resume for user-launched queues.
