import mediapipe as mp
import cv2

class HandTracker:
    #Initializes mediapipe tools
    def __init__(self):
        self.hands = mp.solutions.hands.Hands()
        self.drawer = mp.solutions.drawing_utils

    def process_frame(self, frame):
        #Flips the image horizontally
        frame = cv2.flip(frame, 1)  # 1 = horizontal flip

        #Converts BRG to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Creates an object with hand landmarks and handedness
        self.results = self.hands.process(rgb)
        return frame, self.results

    def display_frame(self, frame, results):
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                self.drawer.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
        

    #Closes the viewing window
    def cleanup(self):
        cv2.destroyAllWindows()
