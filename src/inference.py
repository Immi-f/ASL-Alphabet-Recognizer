"""Real-time ASL letter recognition from webcam using OpenCV."""

import argparse

import cv2
import numpy as np
import torch
from torchvision import transforms

from dataset import LABEL_MAP, IMAGENET_MEAN, IMAGENET_STD
from model import build_model

# Mirror val_transform exactly
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time ASL inference from webcam.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument("--roi-size", type=int, default=300, help="ROI square side length in pixels.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(num_classes=len(LABEL_MAP), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    roi = args.roi_size
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        x1, y1 = cx - roi // 2, cy - roi // 2
        x2, y2 = x1 + roi, y1 + roi

        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop, convert BGR→RGB, preprocess
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = preprocess(crop_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(1)

        label = IDX_TO_LABEL[pred_idx.item()]
        confidence = conf.item()

        text = f"{label} ({confidence:.1%})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("ASL Recognizer", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
