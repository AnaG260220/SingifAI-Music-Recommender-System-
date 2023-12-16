import mediapipe as mp
import numpy as np
import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)

# Prompt the user for the data name
data_name = input("Enter the name of the data: ")

# Initialize mediapipe solutions
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holistic_model = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize lists for storing data and set data size limit
data_list = []
data_size = 0
max_data_size = 150

while True:
    frame_list = []

    _, frame = cap.read()

    # Flip the frame horizontally for a more intuitive view
    frame = cv2.flip(frame, 1)

    # Process the frame using the holistic model
    results = holistic_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.face_landmarks:
        for landmark in results.face_landmarks.landmark:
            # Normalize landmarks relative to a reference point (landmark[1])
            frame_list.extend([landmark.x - results.face_landmarks.landmark[1].x, landmark.y - results.face_landmarks.landmark[1].y])

    # Process left hand landmarks
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            # Normalize landmarks relative to a reference point (landmark[8])
            frame_list.extend([landmark.x - results.left_hand_landmarks.landmark[8].x, landmark.y - results.left_hand_landmarks.landmark[8].y])
    else:
        # If left hand landmarks are not detected, fill with zeros
        frame_list.extend([0.0] * 42)

    # Process right hand landmarks
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            # Normalize landmarks relative to a reference point (landmark[8])
            frame_list.extend([landmark.x - results.right_hand_landmarks.landmark[8].x, landmark.y - results.right_hand_landmarks.landmark[8].y])
    else:
        # If right hand landmarks are not detected, fill with zeros
        frame_list.extend([0.0] * 42)

    # Add the frame data to the list
    data_list.append(frame_list)
    data_size += 1

    # Draw landmarks on the frame
    drawing.draw_landmarks(frame, results.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frame, results.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frame, results.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display data size on the frame
    cv2.putText(frame, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("window", frame)

    # Exit the loop if 'Esc' key is pressed or data size limit is reached
    if cv2.waitKey(1) == 27 or data_size >= max_data_size:
        cv2.destroyAllWindows()
        cap.release()
        break

# Save the collected data to a numpy file
np.save(f"{data_name}.npy", np.array(data_list))
print("Data saved with shape:", np.array(data_list).shape)
