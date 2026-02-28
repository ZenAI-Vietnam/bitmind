# val_infer.py
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_video_val import ValDataset
from model import load_model
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, loader, device="cuda"):
    model.eval()

    total_seen = 0
    valid_seen = 0
    correct = 0
    total_loss = 0.0

    per_ds = defaultdict(lambda: {"n": 0, "correct": 0, "loss": 0.0})
    current_ds = None

    for batch in tqdm(loader, desc="Evaluating"):
        x, y, dataset_name, w = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True).float()

        # log chuyển dataset (đọc dễ nhất khi dataset đã group_by_dataset=True)
        for ds in dataset_name:
            ds = str(ds).strip()
            if ds != "dummy" and ds != current_ds:
                current_ds = ds
                print(f"[VAL] now evaluating dataset: {current_ds}")

        logits = model(x)  # (B,2)

        loss_vec = F.cross_entropy(logits, y, reduction="none")  # (B,)
        mask = w > 0.5

        total_seen += y.size(0)

        if mask.any():
            pred = logits.argmax(dim=1)
            correct += (pred[mask] == y[mask]).sum().item()
            total_loss += loss_vec[mask].sum().item()
            valid_seen += int(mask.sum().item())

            # per-dataset (only valid)
            for i in range(len(dataset_name)):
                if mask[i]:
                    ds = str(dataset_name[i]).strip()
                    per_ds[ds]["n"] += 1
                    per_ds[ds]["correct"] += int(pred[i].item() == y[i].item())
                    per_ds[ds]["loss"] += float(loss_vec[i].item())

    acc = correct / valid_seen if valid_seen > 0 else 0.0
    avg_loss = total_loss / valid_seen if valid_seen > 0 else 0.0
    valid_ratio = valid_seen / total_seen if total_seen > 0 else 0.0

    per_ds_acc = {ds: v["correct"] / v["n"] for ds, v in per_ds.items() if v["n"] > 0}

    return {
        "total_seen": total_seen,
        "valid_seen": valid_seen,
        "valid_ratio": valid_ratio,
        "acc": acc,
        "loss": avg_loss,
        "per_dataset_acc": per_ds_acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="microsoft/xclip-base-patch16-16-frames")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit_per_dataset", type=int, default=322)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no_group", action="store_true", help="disable grouping by dataset_name")
    ap.add_argument("--quiet", action="store_true", help="less dataset indexing logs")
    args = ap.parse_args()

    ds = ValDataset(
        limit_per_dataset=args.limit_per_dataset,
        verbose=(not args.quiet),
        group_by_dataset=(not args.no_group),
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = load_model(
        weights_path=args.weights,
        model_name=args.model_name,
        num_classes=2,
    ).to(args.device)

    metrics = evaluate(model, loader, device=args.device)

    print("\n========== SUMMARY ==========")
    print(f"Total seen:  {metrics['total_seen']}")
    print(f"Valid seen:  {metrics['valid_seen']}  (valid_ratio={metrics['valid_ratio']:.4f})")
    print(f"VAL acc:     {metrics['acc']:.4f}")
    print(f"VAL loss:    {metrics['loss']:.4f}")

    print("\nPer-dataset acc:")
    for ds_name in sorted(metrics["per_dataset_acc"].keys()):
        print(f"  - {ds_name}: {metrics['per_dataset_acc'][ds_name]:.4f}")


if __name__ == "__main__":
    main()