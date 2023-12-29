import cv2
import numpy as np
import time
import HandTrackingModule as htm
import math
import osascript

# Future improvements:
# 1) cancel distance from camera as a factor by using the ratio between palm
# heel and index digit base as a multiplier to the index-tip/thumb-tip distance.

##############################
wCam, hCam = 640, 480
##############################
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

# volResult = osascript.osascript('get volume settings')
# volInfo = volResult[1].split(',')
# outputVol = volInfo[0].replace('output volume:', '')
#volSet = osascript.osascript('set volume output volume 22')
# get current system volume
volCurResult = osascript.osascript('get volume settings')
volCurInfo = volCurResult[1].split(',')
outputCurVol = volCurInfo[0].replace('output volume:', '')
#print ('current volume:', outputCurVol)
#quit()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    vx1, vy1 = 50, 150
    vx2, vy2 = 85, 400

    volPct = outputCurVol
    outputVol = 0
    barVol = np.interp(outputCurVol, [1, 100], [400, 150])

    if len(lmList):
        # print (lmList[4])
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        # thumb tip
        cv2.circle(img, (x1, y1), 7, (255, 150, 75), cv2.FILLED)
        # index tip
        cv2.circle(img, (x2, y2), 7, (255, 150, 75), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 150, 75), 3)
        # center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 5, (255, 150, 75), cv2.FILLED)
        cv2.circle(img, (cx, cy), 2, (255, 255, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        #print(length)

        # Hand range : 12 - 220
        # Vol range : 0 - 99

        outputVol = np.interp(length, [12, 220], [0, 99])
        if outputVol != outputCurVol:
            #print('trying to set volume:',int(outputVol))
            outputCurVol = outputVol
            osascript.osascript("set volume output volume " + str(int(outputVol)))

        if len(lmList):
            barVol = np.interp(length, [12, 220], [400, 150])

        volPct = np.interp(length, [12, 220], [0, 100])
        # volSet = osascript.osascript("set volume " + str(outputVol))

        if length < 20:
            cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

    # VOL label
    cv2.putText(img, 'VOL', (53, 140), 1, 1, (255, 255, 255), 2, cv2.FONT_HERSHEY_SIMPLEX)
    # percent label
    cv2.putText(img, f'%{int(volPct)}', (50, 420), 1, 1, (0, 255, 0), 1, cv2.FONT_HERSHEY_SIMPLEX)

    # draw volume indicator
    #print(barVol)
    cv2.rectangle(img, (50, int(barVol)), (85, 400), (255, 255, 255), cv2.FILLED)

    # draw volume indicator border
    cv2.rectangle(img, (vx1, vy1), (vx2, vy2), (0, 255, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 75, 75), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
