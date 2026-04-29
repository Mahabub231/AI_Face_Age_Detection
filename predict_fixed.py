import json
import os

import cv2
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_opencv")
CROP_DIR = os.path.join(BASE_DIR, "static", "uploads", "crops")

AGE_PROTO = os.path.join(MODEL_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODEL_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODEL_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODEL_DIR, "gender_net.caffemodel")
FACE_PROTO = os.path.join(MODEL_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(MODEL_DIR, "opencv_face_detector_uint8.pb")

AGE_LIST = ["0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100"]
GENDER_LIST = ["Male", "Female"]
AGE_MAP = {
    "0-2": 1,
    "4-6": 5,
    "8-12": 10,
    "15-20": 18,
    "25-32": 28,
    "38-43": 40,
    "48-53": 50,
    "60-100": 70,
}
MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

COLORS = [
    (0, 220, 255), (255, 80, 80), (80, 255, 120), (0, 165, 255),
    (220, 0, 255), (180, 255, 0), (255, 220, 0), (0, 180, 255),
    (255, 128, 0), (128, 0, 255),
]

MAX_FACES = 20
_DNN_SIZES = [300, 500, 700]
_DNN_THRESH = 0.55

_dnn_ok = False
_age_net = None
_gender_net = None
_face_net = None


def _load_dnn():
    global _dnn_ok, _age_net, _gender_net, _face_net

    required = [AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL, FACE_PROTO, FACE_MODEL]

    if not all(os.path.exists(path) for path in required):
        print("[WARN] OpenCV DNN model files missing.")
        return

    try:
        _age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
        _gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
        _face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
        _dnn_ok = True
    except cv2.error as error:
        print(f"[WARN] OpenCV DNN load error: {error}")


_load_dnn()


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union else 0.0


def _nms(raw, iou_thr=0.30):
    raw = sorted(raw, key=lambda item: item[0], reverse=True)
    kept = []

    for box in raw:
        if all(_iou(box[1:], saved[1:]) < iou_thr for saved in kept):
            kept.append(box)

    return kept


def _is_valid_face_box(x1, y1, x2, y2, img_w, img_h):
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return False

    if box_w < 25 or box_h < 25:
        return False

    ratio = box_w / float(box_h)
    if not 0.60 <= ratio <= 1.50:
        return False

    area_ratio = (box_w * box_h) / float(img_w * img_h)
    if area_ratio > 0.40:
        return False

    return True


def _detect_dnn_multiscale(img):
    h, w = img.shape[:2]
    raw = []

    for size in _DNN_SIZES:
        blob = cv2.dnn.blobFromImage(
            img,
            1.0,
            (size, size),
            [104, 117, 123],
            swapRB=False,
        )

        _face_net.setInput(blob)
        detections = _face_net.forward()

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < _DNN_THRESH:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            if not _is_valid_face_box(x1, y1, x2, y2, w, h):
                continue

            raw.append((confidence, x1, y1, x2, y2))

    kept = _nms(raw, iou_thr=0.30)
    return [(x1, y1, x2, y2, conf) for conf, x1, y1, x2, y2 in kept]


def _classify_face(crop):
    blob = cv2.dnn.blobFromImage(
        crop,
        1.0,
        (227, 227),
        MODEL_MEAN,
        swapRB=False,
    )

    _gender_net.setInput(blob)
    gender_predictions = _gender_net.forward()
    gender = GENDER_LIST[int(gender_predictions[0].argmax())]

    _age_net.setInput(blob)
    age_predictions = _age_net.forward()
    age_index = int(age_predictions[0].argmax())
    age_group = AGE_LIST[age_index]

    return gender, age_group, AGE_MAP[age_group]


def predict_image(image_path, annotated_output_path=None, report_output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    os.makedirs(CROP_DIR, exist_ok=True)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    h, w = img.shape[:2]

    min_dim = min(h, w)
    if min_dim < 400:
        scale = 800 / min_dim
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )
        h, w = img.shape[:2]

    detected = _detect_dnn_multiscale(img) if _dnn_ok else []

    detected = sorted(
        detected,
        key=lambda x: ((x[2] - x[0]) * (x[3] - x[1])) * x[4],
        reverse=True,
    )
    detected = detected[:MAX_FACES]

    faces = []
    numeric_ages = []

    for idx, (x1, y1, x2, y2, det_conf) in enumerate(detected):
        pad = max(8, int((x2 - x1) * 0.12))

        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(w - 1, x2 + pad)
        py2 = min(h - 1, y2 + pad)

        padded = img[py1:py2, px1:px2]
        raw_crop = img[y1:y2, x1:x2]

        if raw_crop.size == 0:
            continue

        face_no = idx + 1
        color = COLORS[(face_no - 1) % len(COLORS)]

        crop_fname = f"crop_{stem}_f{face_no}.jpg"
        crop_path = os.path.join(CROP_DIR, crop_fname)
        cv2.imwrite(crop_path, raw_crop)

        crop_url = f"/static/uploads/crops/{crop_fname}"

        if _dnn_ok and padded.size > 0:
            try:
                gender, age_group, age_val = _classify_face(padded)
            except cv2.error:
                gender, age_group, age_val = "Unknown", "N/A", 0
        else:
            gender, age_group, age_val = "Unknown", "N/A", 0

        numeric_ages.append(age_val)

        thick = max(1, int(min(w, h) / 400))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick + 1)

        label = f"#{face_no} {gender}, {age_group}"
        font_scale = max(0.40, min(0.60, w / 1200))
        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            1,
        )

        label_y = max(y1 - 4, text_h + 6)
        cv2.rectangle(
            img,
            (x1, label_y - text_h - 5),
            (x1 + text_w + 8, label_y + 3),
            color,
            -1,
        )
        cv2.putText(
            img,
            label,
            (x1 + 4, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )

        faces.append(
            {
                "face": face_no,
                "age": age_val,
                "age_group": age_group,
                "gender": gender,
                "confidence": round(det_conf * 100, 1),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "crop_url": crop_url,
            }
        )

    if annotated_output_path:
        cv2.imwrite(annotated_output_path, img)

    if faces:
        avg_age = round(sum(numeric_ages) / len(numeric_ages), 1)
        males = sum(1 for face in faces if face["gender"] == "Male")
        females = sum(1 for face in faces if face["gender"] == "Female")
        unknowns = len(faces) - males - females

        parts = []
        if males:
            parts.append(f"Male: {males}")
        if females:
            parts.append(f"Female: {females}")
        if unknowns:
            parts.append(f"Unknown: {unknowns}")

        gender_summary = ", ".join(parts)
        msg = f"{len(faces)} face(s) detected — {gender_summary} [OpenCV DNN]"
    else:
        avg_age = 0.0
        gender_summary = "No Face"
        msg = "No face detected. Try a clearer or closer photo."

    return {
        "predicted_age": avg_age,
        "age_group": faces[0]["age_group"] if faces else "N/A",
        "gender": gender_summary,
        "confidence": 0.0,
        "emotion": "",
        "face_count": len(faces),
        "faces": faces,
        "face_details": json.dumps(faces),
        "message": msg,
        "mode": "opencv",
        "detector": "OpenCV DNN",
    }