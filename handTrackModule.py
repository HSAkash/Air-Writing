import cv2
import mediapipe as mp
import time
import math


class HandTrackModule:
    def __init__(
        self,
        mode=False,
        max_hands=2,
        complexity=1,
        detection_con=0.80,
        tracking_con=0.5
    ):
        
        self.static_image_mode = mode
        self.max_num_hands = max_hands
        self.model_complexity = complexity
        self.min_detection_confidence = detection_con
        self.min_tracking_confidence = tracking_con


        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.static_image_mode,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, frame, draw=False):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)
        if draw:
            if self.results.multi_hand_landmarks is not None:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, hands='right', draw=False):
        self.lmDict = {}
        xList = []
        yList = []
        bbox = []
        hands = hands.lower()
        if self.results.multi_handedness:
            hand_idx = None
            for i, hand in enumerate(self.results.multi_handedness):
                
                if hand.classification[0].label.lower() == hands:
                    hand_idx = i
                    break
            if hand_idx is not None:
                hands = self.results.multi_hand_landmarks[hand_idx]
                for id, lm in enumerate(hands.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    xList.append(cx)
                    yList.append(cy)
                    self.lmDict[id] = (cx, cy)
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                if draw:
                    cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),(0, 255, 0), 2)
        
        return self.lmDict, bbox

    def fingerUp(self, hands='right'):
        fingers = []
        if self.lmDict[self.tipIds[0]][0] < self.lmDict[self.tipIds[0]-1][0]+10 and \
            hands.lower()=='right':
            fingers.append(1)
        elif self.lmDict[self.tipIds[0]][0] > self.lmDict[self.tipIds[0]-1][0]-10 and \
            hands.lower()=='left':
            fingers.append(1)
        else:
            fingers.append(0)
        for i in range(1, len(self.tipIds)):
            if self.lmDict[self.tipIds[i]][1] < self.lmDict[self.tipIds[i]-2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def thumsDown(self):
        thums_down = False
        if self.lmDict[self.tipIds[0]][1] > self.lmDict[0][1] and \
            self.lmDict[self.tipIds[0]][1] > self.lmDict[1][1] and \
            self.lmDict[self.tipIds[0]][1] > self.lmDict[2][1]:
            thums_down = True

        return thums_down

    def findDistance(self, p1, p2, frame, draw=True,r=15, t=3):
        x1, y1 = self.lmDict[p1]
        x2, y2 = self.lmDict[p2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]




def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    print(cv2.CAP_PROP_FRAME_HEIGHT)
    detector = HandTrackModule()
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        

        frame = detector.findHands(frame, draw=True)
        lmList, bbox = detector.findPosition(frame, draw=True, hands='right')
        # if lmList:
        #     print(lmList)
            # quit(1)
        if lmList:
            # print(fingers)
            fingers = detector.fingerUp(hands='right')
            detector.findDistance(4,8,frame, draw=True)
            # print(detector.thumsDown())
            # quit(1)


        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame,
            str(int(fps)),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, 
            (0, 0, 255),
            2
        )


        if not ret:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    main()