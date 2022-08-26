import cv2
import mediapipe as mp
import time
from handTrackModule import HandTrackModule
import numpy as np


class AirWriting:
    def __init__(self,
        #  camera 
        camera_indx=0,
        IMG_WIDTH=640,
        IMG_HEIGHT=480,

        # HandTrackModule & mediapipe
        mode=False,
        max_hands=2,
        complexity=1,
        detection_con=0.80,
        tracking_con=0.5,
        hands='right',
        draw=False,

        # Canvas
        brush_color=(255, 0, 255),
        brush_thickness=15,
        eraser_color=(0, 0, 0),
        eraser_thickness=50
    ):
        self.camera_indx = camera_indx
        
        # HandTrackModule variables
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_con = detection_con
        self.tracking_con = tracking_con
        self.hands = hands

        # AirWriting variables
        self.brush_color = brush_color
        self.brush_thickness = brush_thickness
        self.eraser_color = eraser_color
        self.eraser_thickness = eraser_thickness
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT
        self.draw = draw

        # FPS variables
        self.pTime = 0
        self.cTime = 0
        self.fps = 0

    def getCamera(self):
        self.cap = cv2.VideoCapture(self.camera_indx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.IMG_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.IMG_HEIGHT)

    def createCanvas(self):
        self.xp = 0
        self.yp = 0
        self.canvas = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), np.uint8)

    def start(self):
        # open cv video capture
        self.getCamera()

        # create canvas
        self.createCanvas()

        # HandTrackModule and mediapipe
        self.detector = HandTrackModule(
            mode=self.mode,
            max_hands=self.max_hands,
            complexity=self.complexity,
            detection_con=self.detection_con,
            tracking_con=self.tracking_con
        )

        # main loop
        while True:
            ret, frame = self.cap.read()

            # flip frame
            frame = cv2.flip(frame, 1)

            # mediapipe find hand object
            frame = self.detector.findHands(frame, draw=self.draw)

            # mediapipe find hand landmarks (0-20) points
            lmdict = self.detector.findPosition(frame, hands=self.hands, draw=self.draw)[0]
            if lmdict:
                # Which finger is up
                fingers = self.detector.fingerUp(hands=self.hands)

                # only index finger up (drawing mode)
                if fingers[1] and [x for x in fingers[2:]]==[0,0,0]:
                    x2, y2 = lmdict[8]
                    # print('Drawing Mode')
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = x2, y2
                    # Show brush on frame
                    cv2.line(frame, (self.xp, self.yp), (x2, y2), self.brush_color, int(self.brush_thickness*1.5))
                    # draw on canvas
                    cv2.line(self.canvas, (self.xp, self.yp), (x2, y2), self.brush_color, self.brush_thickness)
                    self.xp, self.yp = x2, y2

                # finger 1 and 2 and 3 up (erase mode)
                elif fingers[1] and fingers[2] and fingers[3]:
                    # get distance from index to middle fingers
                    length, frame, points = self.detector.findDistance(8, 12, frame, draw=False)
                    x2, y2 = lmdict[8]
                    # print('Eraser Mode')
                    if self.xp == 0 and self.yp == 0:
                        self.xp, self.yp = x2, y2
                    # Eraser ractangel shape
                    cv2.rectangle(frame, (points[0], points[1]), (points[2], points[3]), (255, 255, 255), self.eraser_thickness)
                    # Eraser line from canvas
                    cv2.line(self.canvas, (self.xp, self.yp), (x2, y2), self.eraser_color, self.eraser_thickness)
                    self.xp, self.yp = x2, y2

                # other wise reset
                else:
                    self.xp, self.yp = 0, 0

            # canvas overlay on frame
            # color channel 3 to 1 (BGR to GRAY)
            imgGray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

            # overlay canvas on frame
            frame = cv2.bitwise_and(frame, imgInv)
            frame = cv2.bitwise_or(frame, self.canvas)

            # FPS
            self.cTime = time.time()
            self.fps = 1 / (self.cTime - self.pTime)
            self.pTime = self.cTime

            # Show FPS on frame
            cv2.putText(frame,
                str(f"FPS : {int(self.fps)}"),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, 
                (0, 0, 255),
                2
            )

            # Display
            cv2.imshow('frame', frame)
            # cv2.imshow('imageCanvas', imageCanvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



if __name__ == '__main__':
    airWriting = AirWriting(
        # camera
        camera_indx=0,
        IMG_WIDTH=640,
        IMG_HEIGHT=480,

        # HandTrackModule
        mode=False,
        max_hands=2,
        complexity=1,
        detection_con=0.80,
        tracking_con=0.5,
        hands='right',

        # canvas
        brush_color=(255, 0, 255),
        brush_thickness=15,
        eraser_color=(0, 0, 0),
        eraser_thickness=50
    )
    airWriting.start()
