import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.model_complexity = 1
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        return lmList

    def getDistance(self, pt1, pt2):
        return (math.sqrt(((pt1[1] - pt2[1]) ** 2) + ((pt1[2] - pt2[2]) ** 2)))

    def findCurledFingers(self, lmList):
        fingerCurls = []
        knuckleAngles = []
        # if hand
        if len(lmList):
            # thumb test
            controlLength = (self.getDistance(lmList[0], lmList[3])) + 20
            testLength = (self.getDistance(lmList[0], lmList[4]))
            if testLength > controlLength:
                fingerCurls.append(1)
            else:
                fingerCurls.append(0)

            # fingers test
            for iVal in range(2, 6):
                # skip by 4, one pass per digit
                i = ((iVal - 1) * 4) + 1

                # curl test is: if fingertip is closer to palm root than initial finger knuckle is
                # to palm root, digit is curled.
                controlLength = (self.getDistance(lmList[0], lmList[i]))
                testLength = (self.getDistance(lmList[0], lmList[i + 3]))
                if testLength > controlLength:
                    fingerCurls.append(1)
                else:
                    fingerCurls.append(0)

        return fingerCurls


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img, True)
        lmList = detector.findPosition(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
