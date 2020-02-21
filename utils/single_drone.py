import numpy as np
import time
import airsim
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2


def calc_centroid(dilated):
    min_prop = 0.00001
    h, w = np.shape(dilated)

    M = cv2.moments(dilated, binaryImage=True)
    if M["m00"] >= min_prop*h*w:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    else:
        return -1, -1


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    while True:
        # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
        responses = client.simGetImages(
            [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        if response is None:
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        else:
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            image_rgba = img1d.reshape(response.height, response.width, 4)
            image_bgr = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGR)
            cv2.imshow("Image", image_bgr)

        image_hue = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        low = np.array([0, 100, 65])
        high = np.array([10, 200, 130])
        th = cv2.inRange(image_hue, low, high)
        dilated = cv2.dilate(th, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        cv2.imshow("Dilated", dilated)
        cv2.waitKey(1)

        cx, cy = calc_centroid(dilated)
        print(cx, cy)

        key = cv2.waitKey(1) & 0xFF
        if (key == 27 or key == ord('q') or key == ord('x')):
            break


if __name__ == "__main__":
    main()
