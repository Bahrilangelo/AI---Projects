import cv2
import mediapipe
import pyttsx3

camera = cv2.VideoCapture(0)

engine = pyttsx3.init()

mpHands = mediapipe.solutions.hands

hands = mpHands.Hands()

mpDraw = mediapipe.solutions.drawing_utils

checkThumbsUp = False

paper, rock, scissors = False, False, False

while True:
    success, img = camera.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    hlms = hands.process(imgRGB)

    height, width, channel = img.shape

    if hlms.multi_hand_landmarks:
        for handlankmarks in hlms.multi_hand_landmarks:
            
            for fingerNum, landmark in enumerate(handlankmarks.landmark):
                positionX, positionY = int(landmark.x * width), int(landmark.y * height)

                cv2.putText(img, str(fingerNum), (positionX, positionY), 
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0,0,0), 1)

                # if fingerNum > 4 and landmark.y < handlankmarks.landmark[2].y:
                #     break

                if fingerNum == 20 and handlankmarks.landmark[4].y > handlankmarks.landmark[5].y and handlankmarks.landmark[8].y > handlankmarks.landmark[11].y and handlankmarks.landmark[12].y > handlankmarks.landmark[16].y and handlankmarks.landmark[16].y > handlankmarks.landmark[11].y and handlankmarks.landmark[20].y > handlankmarks.landmark[14].y:
                    checkThumbsUp = True

            mpDraw.draw_landmarks(img, handlankmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Camera", img)

    if checkThumbsUp == True:
        engine.say('Paper')
        engine.runAndWait()
        break

    kINP = cv2.waitKey(1)

    if kINP == ord("q"):
        break

cv2.destroyAllWindows()
