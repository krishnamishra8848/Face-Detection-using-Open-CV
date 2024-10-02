import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile  # Add this import for handling video files

# Load pre-trained face detection model (Haar Cascade for face detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return image, len(faces)

# Streamlit app setup
st.title("Face Detection App")

# Choose between uploading an image or a video
option = st.selectbox("Choose the type of media", ["Image", "Video"])

if option == "Image":
    st.subheader("Upload an Image")
    
    # Upload an image file
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Detect faces in the image
        detected_image, num_faces = detect_faces(image)
        
        # Display results
        st.image(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), caption=f"Detected {num_faces} faces", use_column_width=True)

elif option == "Video":
    st.subheader("Upload a Video")
    
    # Upload a video file
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Create a temporary file to store the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        # Use OpenCV to process the video
        cap = cv2.VideoCapture(tfile.name)
        
        stframe = st.empty()  # Placeholder for video
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect faces in the current frame
            detected_frame, _ = detect_faces(frame)
            
            # Display the frame
            stframe.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB), channels="RGB")
        
        cap.release()
