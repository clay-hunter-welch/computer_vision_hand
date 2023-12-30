import cv2
import HandTrackingModule as htm
import numpy as np
import math
import time
import os

xOff, yOff = 15, 200
xAnimOff, yAnimOff = 25, 170
# import finger images
folderPath = "FingerCountingImages"

backdrop = cv2.imread(f'{folderPath}/backdrop.png', cv2.IMREAD_UNCHANGED)
arm = cv2.imread(f'{folderPath}/arm.png', cv2.IMREAD_UNCHANGED)
palm = cv2.imread(f'{folderPath}/palm.png', cv2.IMREAD_UNCHANGED)
thumbsUp = cv2.imread(f'{folderPath}/thumbsUp.png', cv2.IMREAD_UNCHANGED)

cList = sorted(os.listdir(f'{folderPath}/closedUp'))
eList = sorted(os.listdir(f'{folderPath}/extended'))

closedUp = []
extended = []

for imgPath in cList:
    image = cv2.imread(f'{folderPath}/closedUp/{imgPath}', cv2.IMREAD_UNCHANGED)
    closedUp.append(image)
for imgPath in eList:
    image = cv2.imread(f'{folderPath}/extended/{imgPath}', cv2.IMREAD_UNCHANGED)
    extended.append(image)

# set up cam read
wCam, hCam = 640, 480
cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector()


def merge_image(back, front, x, y):
    # convert to rgba
    if back.shape[2] == 3:
        back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
    if front.shape[2] == 3:
        front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

    # crop the overlay from both images
    bh, bw = back.shape[:2]
    fh, fw = front.shape[:2]
    x1, x2 = max(x, 0), min(x + fw, bw)
    y1, y2 = max(y, 0), min(y + fh, bh)
    front_cropped = front[y1 - y:y2 - y, x1 - x:x2 - x]
    back_cropped = back[y1:y2, x1:x2]

    alpha_front = front_cropped[:, :, 3:4] / 255
    alpha_back = back_cropped[:, :, 3:4] / 255

    # replace an area in result with overlay
    result = back.copy()
    # print(
    #     f'af: {alpha_front.shape}\nab: {alpha_back.shape}\nfront_cropped: {front_cropped.shape}\nback_cropped: {back_cropped.shape}')
    result[y1:y2, x1:x2, :3] = alpha_front * front_cropped[:, :, :3] + (1 - alpha_front) * back_cropped[:, :, :3]
    result[y1:y2, x1:x2, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front * alpha_back) * 255

    return result


def main():
    bobFactor = 0
    weaveFactor = 0
    while True:
        fingerState = [0, 0, 0, 0, 0]
        bobFactorRaw = (math.sin(2.3 * time.time())) * 5
        bobFactor = int(bobFactorRaw)
        weaveFactorRaw = (math.sin(1.5 * time.time())) * 3
        weaveFactor = int(weaveFactorRaw)
        thumbsUpOrientation = 0

        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList):
            fingerState, thumbsUpOrientation = detector.findCurledFingers(lmList)
        # if sum(fingerState) == 1 and fingerState[0]:
        #    thumbsUpFlag = 1

        img = merge_image(img, backdrop, xOff, yOff)
        img = merge_image(img, arm, xAnimOff + weaveFactor, yAnimOff + bobFactor + 30)

        if sum(fingerState) == 1 and fingerState[0] and thumbsUpOrientation:
            img = merge_image(img, thumbsUp, xAnimOff + weaveFactor, yAnimOff + bobFactor)
        else:
            img = merge_image(img, palm, xAnimOff + weaveFactor, yAnimOff + bobFactor)

            # add fingers to match fingerState
            for digit in range(1, 5):
                if fingerState[digit]:
                    img = merge_image(img, extended[digit], xAnimOff + weaveFactor, yAnimOff + bobFactor)
                else:
                    img = merge_image(img, closedUp[digit], xAnimOff + weaveFactor, yAnimOff + bobFactor)
            # add thumb to match fingerstate
            if fingerState[0]:
                img = merge_image(img, extended[0], xAnimOff + weaveFactor, yAnimOff + bobFactor)
            else:
                img = merge_image(img, closedUp[0], xAnimOff + weaveFactor, yAnimOff + bobFactor)

        cv2.imshow("oooOOOOooooOOOOooo!", img)
        cv2.waitKey(1)


main()
