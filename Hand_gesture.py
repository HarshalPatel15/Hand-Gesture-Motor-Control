#!/usr/bin/env python3
import cv2
import mediapipe as mp
import rospy
from std_msgs.msg import Float32
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import threading
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import test_train_split
from sklearn.linear_model import LinearRegression


data_1=pd.read_csv(r"/home/aman/Downloads/Training_dataset.csv")
x=data_1.iloc[:,0].values
y=data_1.iloc[:,1].values

x_train,x_test,y_train,y_test=test_train_split(x,y,test_size=0.05)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)



# Global variables for storing hand percentage
percentage = 0

# Function for capturing video and hand tracking
def capture_video():
    global percentage
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    percentage_pub = rospy.Publisher('mega', Float32, queue_size=1)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Convert the frame to RGB for Mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

            # Calculate the distance between index finger and thumb
            length = ((index_finger_tip.x - thumb_tip.x) ** 2 + (index_finger_tip.y - thumb_tip.y) ** 2) ** 0.5

            # Calculate the percentage within the range [0, 100]
            hand_length = ((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x - hand_landmarks.landmark[
                mp_hands.HandLandmark.THUMB_TIP].x) ** 2 + (
                                   hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y - hand_landmarks.landmark[
                               mp_hands.HandLandmark.THUMB_TIP].y) ** 2) ** 0.5
            percentage = min(max(length / hand_length * 100, 0), 100)

            percentage_pub.publish(percentage)

            # Display percentage on the frame
            cv2.putText(frame, f'Percentage: {percentage:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        2)

        cv2.imshow('Hand Length Percentage', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


classifier=LinearRegression()
classifier.fit(x_train,y_train)
y_pred=LinearRegression.predict(x_test)

# Function for plotting the graph
def plot_graph():
    global percentage
    while True:
        plt.scatter(percentage, percentage, color='red')
        plt.plot(percentage, percentage, color='blue')  # Plotting the regression line
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.title('Linear Regression: Salary vs Experience')
        plt.show(block=False)
        plt.pause(0.001)

# Initialize ROS node
rospy.init_node('mega')

# Create and start the threads
video_thread = threading.Thread(target=capture_video)
plot_thread = threading.Thread(target=plot_graph)
video_thread.start()
plot_thread.start()
