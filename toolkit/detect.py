from insightface.app import FaceAnalysis
from .align import alignment_procedure

app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)

def detect_faces(img):
    faces = app.get(img)
    result = []
    for face in faces:
        bbox = face.bbox
        landmarks = face.kps
        result.append({
            'bbox': bbox.tolist(),
            'left_eye': landmarks[0].tolist(),
            'right_eye': landmarks[1].tolist(),
            'score': face['det_score']
        })
    return sorted(result, key=lambda x: x['score'], reverse=True)


def extract_face(img, detections):
    results = []
    for detection in detections:
        x1, y1, x2, y2 = list(map(lambda x: int(x), detection["bbox"]))

        detected_face = img[y1:y2, x1:x2]
        results.append({
            'cropped_face': detected_face,
            'left_eye': detection['left_eye'],
            'right_eye': detection['right_eye']
        })
    return results


def detect_crop_align(image, k=1):
    detections = detect_faces(image)
    extractions = extract_face(image, detections)
    
    aligned_images = []
    for item in extractions:
        aligned_img, _ = alignment_procedure(
            item['cropped_face'],
            {
                'left_eye': item['left_eye'],
                'right_eye': item['right_eye']
            }
        )
        aligned_images.append(aligned_img)
    return aligned_images[:k]