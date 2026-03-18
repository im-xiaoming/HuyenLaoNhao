import cv2
from retinaface import RetinaFace

def detect_faces(img):
    detections = RetinaFace.detect_faces(img)

    if not isinstance(detections, dict):
        return []

    results = []

    for key in detections:
        face = detections[key]

        if face["score"] > 0.90:
            results.append(face)

    return sorted(results, key=lambda x: x['score'], reverse=True)

def extract_face(img, detection):
    x1, y1, x2, y2 = detection["facial_area"]

    detected_face = img[int(y1):int(y2), int(x1):int(x2)]

    return detected_face, detection['landmarks'], (x1, y1, x2 - x1, y2 - y1)
