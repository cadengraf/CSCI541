"""Training signatures, best-model saves, and resumable full state."""

import hashlib
import json
import random

import torch

from .model import EMBEDDING_SIZE


def training_signature(args):
    """Reject recovery when settings that affect training have changed."""
    keys = ("sop_root", "abo_root", "voc_root", "fashion_root", "abo_max_items",
            "source", "epochs", "batch_size", "instances_per_id", "steps_per_epoch",
            "source_batch_weights", "triplet_margin", "triplet_weight", "mask_every",
            "mask_loss_weight", "learning_rate", "seed", "augmentation",
            "reid_only", "selection", "resume")
    settings = {key: getattr(args, key, None) for key in keys}
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()


def recovery_path(checkpoint):
    return checkpoint.with_name(checkpoint.stem + ".last.pt")


def save_recovery(path, model, classifier, optimizer, identity_optimizer,
                  model_schedule, identity_schedule, epoch, baseline, best_score,
                  history, signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    torch.save({"model": model.state_dict(), "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "identity_optimizer": identity_optimizer.state_dict(),
                "model_schedule": model_schedule.state_dict(),
                "identity_schedule": identity_schedule.state_dict(),
                "epoch": epoch, "baseline": baseline, "best_score": best_score,
                "history": history, "signature": signature,
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, temporary)
    temporary.replace(path)

def save_joint_checkpoint(model, args, epoch, selection_metrics, selection_score, instances_per_id):
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_name(args.checkpoint.name + ".partial")
    torch.save({"model": model.state_dict(), "embedding_size": EMBEDDING_SIZE,
                "use_mask_pool": model.use_mask_pool,
                "segmentation_trained": not args.reid_only,
                "training": {"loss": "identity_ce+batch_hard_triplet",
                             "identities_per_batch": args.batch_size // instances_per_id,
                             "instances_per_id": instances_per_id, "epoch": epoch,
                             "start_checkpoint": str(args.resume) if args.resume else None,
                             "signature": training_signature(args),
                             "selection_metrics": selection_metrics,
                             "selection_score": selection_score}}, temporary)
    temporary.replace(args.checkpoint)
