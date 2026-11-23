# Active local datasets

| Dataset | Local location | Role |
|---|---|---|
| VRAI | `data/aerial/vrai/` | Aerial vehicle identity training and local held-out retrieval; official test images are not present here. |
| VRU | `data/aerial/vru/` | Aerial training and full fixed-gallery test; Large has 83,595 queries and 8,000 gallery identities. |
| Prepared mixed data | `data/aerial_training/` | Audited/namespaced VRAI/VRU training and development manifests. Preserve these exact files for resume. |
| VisDrone MOT | `data/aerial/VisDrone2019-MOT-{train,val}/` | Within-video aerial identities. They do not supply cross-camera correspondence. |
| KITTI MOTS | `data/kitti_mots/` | Existing ground-video car detection/mask/tracking evaluation. |
| CityFlowV2 | `/home/megrad/Downloads/AICity22_Track1_MTMC_Tracking/` | Complete ground-camera validation: 23 cameras / 81,954 frames, supplied detections and official MTMC evaluator. |
| iSAID | `data/aerial/isaid/` | Instance masks, not physical ReID identity labels. |

Source Downloads are read-only inputs. Do not move datasets or rewrite existing manifests during cleanup. Existing checkpoints and evaluation signatures reference their current paths.

`prepare.py --help` lists the current preparation entry points. Broader dataset notes and older object/person workflows are retained in [archive/aerial_data_notes.md](archive/aerial_data_notes.md). Numeric IDs in separate drone/infrastructure annotation exports need verified correspondence before being used as cross-view truth.

Train/development/test exposure, gallery choice and available camera labels must be explicit. Published-result comparisons do not require retraining authors' baselines or obtaining their exact gallery choice. The user launches all preparation jobs involving downloads or long processing runs.
