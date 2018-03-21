import dlib
import face_landmarks

_detector = dlib.get_frontal_face_detector()


def detect(image, min_score=0.2):
    boxes, scores, idx = _detector.run(image, 1, -1)

    output = []
    for i, d in enumerate(boxes):
        if scores[i] < min_score:
            continue

        x, y, w, h = d.left(), d.top(), d.width(), d.height()
        face = {
            "box": [x, y, w, h],
            "score": scores[i],
            "index": idx[i],
            "shape": face_landmarks.face_shape(image, d),
        }
        output.append(face)

    return output
