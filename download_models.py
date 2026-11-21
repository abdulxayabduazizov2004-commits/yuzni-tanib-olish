"""OpenCV Zoo'dan kerakli ONNX modellarni models/ papkasiga yuklab oladi."""
import sys
import urllib.request
from pathlib import Path

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELS = {
    "face_detection_yunet_2023mar.onnx": "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "person_reid_youtu_2021nov.onnx": "person_reid_youtureid/person_reid_youtu_2021nov.onnx",
}
CLIP = "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/"
URLS = {name: BASE + rel for name, rel in MODELS.items()} | {
    "clip_vision_model_quantized.onnx": CLIP + "onnx/vision_model_quantized.onnx",
    "clip_text_model_quantized.onnx": CLIP + "onnx/text_model_quantized.onnx",
    "clip_tokenizer.json": CLIP + "tokenizer.json",
}

out = Path(__file__).resolve().parent / "models"
out.mkdir(exist_ok=True)
for name, url in URLS.items():
    dst = out / name
    if dst.is_file() and dst.stat().st_size > 100_000:
        print(f"bor: {name}")
        continue
    print(f"yuklanmoqda: {name} ...", flush=True)
    try:
        urllib.request.urlretrieve(url, dst)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Xato: {name} yuklanmadi: {e}")
print("Tayyor.")
