"""
download_opencv_models.py
Run this once to download the pre-trained OpenCV Caffe models.
These are included in the repo already — only run if models are missing.
"""
import urllib.request
import os
from pathlib import Path

MODEL_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "models_opencv"
MODEL_DIR.mkdir(exist_ok=True)

URLS = {
    "opencv_face_detector.pbtxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb":
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb",
    "age_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt",
    "age_net.caffemodel":
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel",
    "gender_deploy.prototxt":
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt",
    "gender_net.caffemodel":
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
}

def download_all():
    for name, url in URLS.items():
        path = MODEL_DIR / name
        if path.exists() and path.stat().st_size > 1024:
            print(f"✅ Already exists: {name}")
            continue
        print(f"⬇️  Downloading: {name} ...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"   ✅ Done ({path.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print("\n✅ All models ready in:", MODEL_DIR)

if __name__ == "__main__":
    download_all()
