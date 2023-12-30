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
        thumbsUpOrientation = 0

        if lmList[4][2] < lmList[9][2]:
            thumbsUpOrientation = 1

        # if hand
        if len(lmList):
            unitLength = self.getDistance(lmList[0], lmList[5])

            # thumb test
            controlLength = (self.getDistance(lmList[0], lmList[3])) + (unitLength/5)
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
                # to palm root, and orientation is not thumbs up, digit is curled.
                #  -or- if fingertip is closer to finger/palm connection than
                # the next knuckle up from connection is to finger/palm connection. Trying to stabilise
                # and catch edge cases.
                controlLength1 = (self.getDistance(lmList[0], lmList[i]))   # palm root to finger connection joint
                testLength1 = (self.getDistance(lmList[0], lmList[i + 3]))  # palm root to finger tip
                controlLength2 = (self.getDistance(lmList[i], lmList[i+1])) # finger connection joint to next joint up
                testLength2 = (self.getDistance(lmList[i], lmList[i+3]))    # finger connection joint to fingertip
                print(controlLength1, testLength1)
                #if (testLength1 > controlLength1) or (testLength2 < controlLength2):
                if ((testLength1 > controlLength1) and not thumbsUpOrientation) or ((testLength2 > controlLength2) and thumbsUpOrientation):
                    fingerCurls.append(1)
                else:
                    fingerCurls.append(0)


        return fingerCurls, thumbsUpOrientation


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
