import cv2
import time
import json
from webcam_video_stream import WebcamVideoStream
from face_detection import face_detector, face_descriptor

print "~ Starting trainer program , To exit press 'q' on window."

camera = WebcamVideoStream(src=0, size_w=320, size_h=240).start()

last_time = -1
samples = []

while True:
    ret, image = camera.read()
    if not ret:
        break

    faces = face_detector.detect(image)
    time_validation = True if last_time == -1 else time.time() - last_time > 5
    valid_for_train = len(faces) == 1 and time_validation

    cv2.rectangle(image, (5, 5), (320 - 5, 240 - 5),
                  (0, 255 if valid_for_train else 0, 0 if valid_for_train else 255), 2)

    if valid_for_train:
        face = faces[0]
        x, y, w, h = face["box"]
        user_face = list(face_descriptor.describe(image, face["shape"]))
        cv2.rectangle(image, (x, y), (x + w, y + h), (100, 200, 100))

    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32 and valid_for_train:  # space key
        last_time = time.time()
        samples.append(user_face)
        print "~ New sample has been added to list"

camera.stop()

if not samples:
    print "~ There is no samples in list"
    exit(1)

with open('user_model.json', 'w') as handle:
    json.dump(samples, handle)

print "~ User information has been saved to user_model.json"
