This project aims to develop an Emotion-Based Music Recommender System using Facial Recognition. 
The system leverages technologies like Streamlit WebRTC module, OpenCV, Keras, and Mediapipe to capture and analyze facial expressions in real-time. 
Basically, the System using facial recognition is designed to enhance the music listening experience by tailoring music recommendations based on the user's emotional state detected through facial expressions. 
The detected emotions are then used to recommend suitable music tracks, providing a personalized and immersive music listening experience.

Objectives
The main objective of this project is to create a system that can accurately recognize the user's emotions through facial expressions and recommend music that aligns with their current emotional state. To sum up the objectives it is as follows :
1.	Utilize facial recognition technology to detect and analyze the user's emotional state.
2.	Integrate real-time video streaming through Streamlit WebRTC module for efficient processing.
3.	Implement a recommendation algorithm that maps detected emotions to suitable music genres or tracks.
4.	Develop a user-friendly interface for seamless interaction.

Technologies Used
1.	Streamlit WebRTC Module: Enables real-time video streaming for facial recognition.
2.	OpenCV (Open Source Computer Vision Library): Used for capturing and processing video frames.
3.	Keras: Utilized for emotion detection through a pre-trained deep learning model.
4.	Mediapipe: Used for facial landmark detection to enhance emotion recognition accuracy.
5.	WebRTC Module: Enables real-time video streaming directly in the browser.

System Architecture
1.	Input Module (WebRTC): Captures real-time video feed from the user's webcam.
2.	Facial Landmark Detection (Mediapipe): Identifies facial features for accurate emotion analysis.
3.	Emotion Recognition (Keras Model): Classifies the detected facial expressions into emotions.
4.	Music Recommendation Algorithm: Maps emotions to suitable music genres or tracks.
5.	User Interface (Streamlit): Provides an interactive platform for the user to interact with the system.

Methodology
•	Face Detection: We use OpenCV for real-time face detection in the video stream provided by the user's webcam.
•	Facial Landmark Detection: Mediapipe is used to identify key facial landmarks (e.g., eyes, mouth) which are crucial for emotion analysis.
•	Emotion Classification: A deep learning model (implemented using Keras) is trained on a dataset of labeled facial expressions to classify emotions.
•	Music Recommendation: Based on the predicted emotion, the system recommends a playlist or specific songs associated with that emotion.
