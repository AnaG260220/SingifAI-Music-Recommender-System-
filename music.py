import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

# Load model and label data
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize MediaPipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Define an extended list of languages and singers
languages = ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean", "Russian", "Other"]
singers = ["Adele", "Ed Sheeran", "Taylor Swift", "Beyoncé", "Justin Bieber", "Arijit Singh", "Shreya Ghoshal", "Atif Aslam", "Lata Mangeshkar", "Kishore Kumar", "Other"]

# Function to process emotions and render video
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]

            cv2.putText(frm, pred, (50, 50),
                        cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy", np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks,
                               holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(
            frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(
            frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

def main():
    st.set_page_config(
        page_title="Emotion-Based Song Recommender",
        layout="wide",
    )

    st.title("Emotion-Based Song Recommender")

    # Place the smaller logo in the top-left corner
    st.sidebar.image("https://i.ibb.co/rZXk36X/logo.png", width=300)

    st.sidebar.header("User Input")

    # Dropdown for choosing language
    lang_option = st.sidebar.selectbox("Choose Language", languages)
    if lang_option == "Other":
        lang = st.sidebar.text_input("Enter Language")
    else:
        lang = lang_option

    # Dropdown for choosing singer
    singer_option = st.sidebar.selectbox("Choose Singer", singers)
    if singer_option == "Other":
        singer = st.sidebar.text_input("Enter Singer")
    else:
        singer = singer_option

    # Button for song recommendation
    btn = st.sidebar.button("Recommend me songs")

    # Session variable to track camera state
    camera_opened = st.session_state.get("camera_opened", False)

    if not camera_opened:
        st.session_state["camera_opened"] = False

    try:
        emotion = np.load("emotion.npy")[0]
    except:
        emotion = ""

    if not emotion:
        st.session_state["run"] = "true"
    else:
        st.session_state["run"] = "false"

    if lang and singer and st.session_state["run"] != "false":
        # Set the camera state to True
        st.session_state["camera_opened"] = True

        # Use CSS to set light and dark mode background colors
        
        st.write(
            f"""
            <style>
            body {{
                background-color: #3498db !important; /* Blue background color */
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            """
            <style>
           
            .stSidebar {
                background-color: #252b42; /* Dark mode sidebar background color */
            }
            .stApp {
                background-color: #ffffff; /* Light mode app background color */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Use CSS to center and adjust the camera preview size
        st.markdown(
            """
            <style>
            .stStream {
                max-width: 50% !important;
                margin: 0 auto;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        webrtc_streamer(key="camera", desired_playing_state=True,
                        video_processor_factory=EmotionProcessor)

    if btn:
        if not emotion:
            st.warning("Please let me capture your emotion first")
            st.session_state["run"] = "true"
        else:   
            webbrowser.open(
                f"https://www.youtube.com/results?search_query={lang}+{emotion}+songs+{singer}")
            np.save("emotion.npy", np.array([""]))
            st.session_state["run"] = "false"

if __name__ == "__main__":
    main()
