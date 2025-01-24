import cv2
import dlib
import time
faces = []


# Load the pre-trained face detector and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(r"C:\Users\sarxx\Downloads\drive-download-20240911T134028Z-001\dataset.dat")

# Open the video file
video_path = r"C:\Users\sarxx\Downloads\drive-download-20240911T134028Z-001\sample.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize variables
ear_threshold = 0.8  # Adjust this value based on your environment and setup
landmark_unavailable_start_time = None
distraction_duration_threshold = 5  # Duration in seconds

while True:
    ret, frame = cap.read()

    if not ret:
        # End of video or error reading frame
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(gray.dtype)  # Should output: uint8



    if len(faces) > 0:
        # Reset landmark unavailable start time when a face is detected
        landmark_unavailable_start_time = None

        for face in faces:
            # Get the facial landmarks for each face
            landmarks = landmark_predictor(gray, face)

            # Extract the refined eye region using the landmark indices
            left_eye = landmarks.part(36).x, landmarks.part(36).y, landmarks.part(39).x, landmarks.part(39).y
            right_eye = landmarks.part(42).x, landmarks.part(42).y, landmarks.part(45).x, landmarks.part(45).y

            # Calculate the eye aspect ratio (EAR)
            # EAR = (|P2 - P6| + |P3 - P5|) / (2 * |P1 - P4|)
            left_ear = ((landmarks.part(39).y - landmarks.part(37).y) + (landmarks.part(40).y - landmarks.part(38).y)) \
                       / (2 * (landmarks.part(41).x - landmarks.part(36).x))
            right_ear = ((landmarks.part(45).y - landmarks.part(43).y) + (landmarks.part(46).y - landmarks.part(44).y)) \
                        / (2 * (landmarks.part(47).x - landmarks.part(42).x))

            # Check if the eyes are closed or open based on EAR threshold
            if left_ear < ear_threshold and right_ear < ear_threshold:
                cv2.putText(frame, 'Eyes Closed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'Eyes Open', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw rectangles around the eyes
            cv2.rectangle(frame, (left_eye[0], left_eye[1]), (left_eye[2], left_eye[3]), (0, 255, 0), 2)
            cv2.rectangle(frame, (right_eye[0], right_eye[1]), (right_eye[2], right_eye[3]), (0, 255, 0), 2)

            # Draw the detected landmarks
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
    else:
        # Facial landmarks not available
        if landmark_unavailable_start_time is None:
            # Start the timer when facial landmarks are first unavailable
            landmark_unavailable_start_time = time.time()
        else:
            # Check if the duration of facial landmarks unavailability exceeds the threshold
            elapsed_time = time.time() - landmark_unavailable_start_time
            if elapsed_time >= distraction_duration_threshold:
                cv2.putText(frame, 'You are distracted', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Optional: Add a small delay to simulate real-time processing
    time.sleep(0.1)  # Adjust the sleep time if needed

# Release the video capture
cap.release()
