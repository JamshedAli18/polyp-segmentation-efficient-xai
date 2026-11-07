# infer.py - run the INT8 polyp segmentation model on one image (CPU, ONNX Runtime)
import argparse

import cv2
import numpy as np
import onnxruntime as ort

S = 352
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

p = argparse.ArgumentParser()
p.add_argument("image")
p.add_argument("--model", default="models/student_mnv2_int8.onnx")
p.add_argument("--out", default="prediction.png")
args = p.parse_args()

bgr = cv2.imread(args.image)
H, W = bgr.shape[:2]
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
x = (cv2.resize(rgb, (S, S), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255 - MEAN) / STD
x = x.transpose(2, 0, 1)[None]

sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
logits = sess.run(None, {"image": x})[0][0, 0]
prob = cv2.resize(1 / (1 + np.exp(-logits)), (W, H), interpolation=cv2.INTER_LINEAR)
mask = (prob > 0.5).astype(np.uint8)

overlay = bgr.copy()
overlay[mask > 0] = (0.5 * overlay[mask > 0] + np.array([0, 0, 127])).astype(np.uint8)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
cv2.imwrite(args.out, overlay)
print(f"polyp area: {mask.mean()*100:.1f}% of image -> saved {args.out}")