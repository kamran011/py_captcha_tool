"""
Train a CNN + BiLSTM + CTC (CRNN) text-CAPTCHA recogniser and export to ONNX.

    python ml/train_crnn.py --epochs 25

Reads data/synthetic/{train,val}, writes data/models/crnn.onnx and
data/models/train_log.json.  Test split is never touched here; use
ml/evaluate.py for the held-out number.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.common import (  # noqa: E402
    BLANK, CHAR_TO_IDX, IMG_H, IMG_W, NUM_CLASSES, ctc_greedy_decode, preprocess,
)

DATA = Path("data/synthetic")
MODELS = Path("data/models")


class CRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def block(i, o, pool):
            layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(pool))
            return layers

        self.cnn = nn.Sequential(
            *block(1, 32, (2, 2)),     # 24 x 80
            *block(32, 64, (2, 2)),    # 12 x 40
            *block(64, 96, (2, 2)),    # 6 x 20
            *block(96, 128, (2, 1)),   # 3 x 20  -> T = 20 frames (>= 2*6 for CTC)
        )
        self.rnn = nn.LSTM(128 * (IMG_H // 16), 128, num_layers=2,
                           bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):                      # x: (B,1,H,W)
        f = self.cnn(x)                        # (B,C,H',W')
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)   # (B,T,C*H')
        out, _ = self.rnn(f)
        return self.fc(out)                    # (B,T,num_classes)


class CaptchaSet(Dataset):
    def __init__(self, split: str):
        folder = DATA / split
        with open(folder / "labels.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.items = [(folder / r["filename"], r["text"]) for r in rows]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, text = self.items[i]
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(preprocess(img)), text


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    texts = [b[1] for b in batch]
    targets = torch.tensor([CHAR_TO_IDX[c] for t in texts for c in t], dtype=torch.long)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    return imgs, targets, lengths, texts


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for imgs, _, _, texts in loader:
        logits = model(imgs.to(device)).cpu().numpy()
        for lg, truth in zip(logits, texts):
            correct += ctc_greedy_decode(lg)[0] == truth
            total += 1
    return correct / max(total, 1)


def export_onnx(model: CRNN, path: Path) -> None:
    model.eval().cpu()
    dummy = torch.zeros(1, 1, IMG_H, IMG_W)
    torch.onnx.export(
        model, dummy, str(path), input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    train_dl = DataLoader(CaptchaSet("train"), args.batch, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(CaptchaSet("val"), 256, collate_fn=collate)

    model = CRNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl))
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    best, log = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum = time.time(), 0.0
        for imgs, targets, lengths, _ in train_dl:
            logits = model(imgs.to(device))                    # (B,T,C)
            logp = logits.log_softmax(2).permute(1, 0, 2)      # (T,B,C)
            in_len = torch.full((logits.size(0),), logits.size(1), dtype=torch.long)
            loss = ctc(logp, targets, in_len, lengths)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            loss_sum += loss.item()
        val_acc = evaluate(model, val_dl, device)
        entry = {"epoch": epoch, "loss": round(loss_sum / len(train_dl), 4),
                 "val_exact": round(val_acc, 4), "secs": round(time.time() - t0, 1)}
        log.append(entry)
        print(entry, flush=True)
        if val_acc > best:
            best = val_acc
            export_onnx(model, MODELS / "crnn.onnx")
            torch.save(model.state_dict(), MODELS / "crnn.pt")
            model.to(device)

    (MODELS / "train_log.json").write_text(json.dumps(log, indent=2))
    print(f"best val exact-match: {best:.4f}  -> {MODELS / 'crnn.onnx'}")


if __name__ == "__main__":
    main()
