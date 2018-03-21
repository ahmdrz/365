import math


def euclidean_distance(instance1, instance2, length):
    distance = 0
    for i in range(length):
        distance += pow((instance1[i] - instance2[i]), 2)
    return math.sqrt(distance)


if __name__ == "__main__":
    import cv2
    import json
    import time
    import os
    from webcam_video_stream import WebcamVideoStream
    from face_detection import face_detector, face_descriptor

    user_model_file = os.path.join(os.path.dirname(__name__), "user_model.json")
    images_path = os.path.join(os.path.dirname(__name__), "images")

    if not os.path.exists(user_model_file):
        print "~ user_model.json not found. Run train_user.py to generate it !"
        exit(1)

    with open(user_model_file, 'r') as handle:
        samples = json.load(handle)

    if not os.path.exists(images_path):
        os.mkdir(images_path, 0755)

    camera = WebcamVideoStream(src=0, size_w=640, size_h=480).start()

    while True:
        ret, image = camera.read()
        if not ret:
            break

        faces = face_detector.detect(image)

        has_user = False
        for face in faces:
            encoding = face_descriptor.describe(image, face["shape"])
            percent = 0
            for sample in samples:
                percent += 1 if euclidean_distance(encoding, sample, 128) < 0.6 else 0
            percent /= float(len(samples))
            percent = int(percent * 100)

            if percent > 90:
                has_user = True
                break

        if has_user:
            image_file = "{}/{}.jpg".format(images_path, int(time.time()))
            print "~ Saving image to {} file".format(image_file)
            cv2.imwrite(image_file, image)
            break

    print "~ Stopping camera ..."
    camera.stop()
