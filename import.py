import cv2
import dlib
import time
import winsound
import numpy as np

# Load the pre-trained face detector and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(r"C:\dataset.dat")

# Load the pre-trained gender detection model (DNN module)
gender_net = cv2.dnn.readNetFromCaffe(
    r'C:\zz my personal\project\drive-download-20240911T134028Z-001\deploy_gender.prototxt', 
    r'C:\zz my personal\project\drive-download-20240911T134028Z-001\gender_net.caffemodel'
)

# Gender classes
gender_classes = ['Male', 'Female']

# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Initialize variables
ear_threshold = 0.5  # Adjust this value based on your environment and setup
landmark_unavailable_start_time = None
distraction_duration_threshold = 2  # Duration in seconds for general distraction alert
eyes_closed_duration_threshold = 2  # Duration in seconds for eye-closure specific beep
eyes_closed_start_time = None  # Track the time when eyes are first detected closed
mar_threshold = 0.6  # Adjust based on the environment to detect yawning
yawn_duration_threshold = 3  # Time threshold to detect a sustained yawn
yawn_start_time = None  # Track the time when yawning starts

# Define a function to play a beep sound
def play_beep():
    frequency = 1000  # Set Frequency To 1000 Hertz
    duration = 500    # Set Duration To 500 ms == 0.5 second
    winsound.Beep(frequency, duration)

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(landmarks):
    # Convert dlib point objects to numpy arrays for mouth coordinates
    top_lip = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(50, 53)])
    bottom_lip = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(65, 68)])
    mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 54)])

    # Calculate vertical distances (Euclidean distance between corresponding points)
    A = np.linalg.norm(bottom_lip[1] - top_lip[1])  # Middle part of the lips
    B = np.linalg.norm(bottom_lip[0] - top_lip[0])  # Left part of the lips
    C = np.linalg.norm(bottom_lip[2] - top_lip[2])  # Right part of the lips

    # Calculate horizontal distance (between corners of the mouth)
    D = np.linalg.norm(mouth[3] - mouth[0])  # Distance between two corners of the mouth

    # MAR is the average of vertical distances divided by horizontal distance
    mar = (A + B + C) / (3.0 * D)
    return mar

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            # Error reading frame from the video capture
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector(gray)

        if len(faces) > 0:
            # Reset landmark unavailable start time when a face is detected
            landmark_unavailable_start_time = None

            for face in faces:
                # Get the facial landmarks for each face
                landmarks = landmark_predictor(gray, face)

                # Extract the face ROI for gender classification
                (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
                face_roi = frame[y:y+h, x:x+w]

                # Prepare the face ROI for gender classification (resize to match model input size)
                blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

                # Pass the face ROI through the gender model
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_classes[gender_preds[0].argmax()]

                # Display gender on the frame
                cv2.putText(frame, f'Gender: {gender}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Draw facial landmarks on the face
                for i in range(68):  # There are 68 landmarks in dlib's shape predictor
                    landmark_x = landmarks.part(i).x
                    landmark_y = landmarks.part(i).y

                    # Use red color for mouth landmarks (48 to 67 are the mouth landmarks)
                    if 48 <= i <= 67:
                        cv2.circle(frame, (landmark_x, landmark_y), 1, (0, 0, 255), -1)  # Red for mouth landmarks
                    else:
                        cv2.circle(frame, (landmark_x, landmark_y), 1, (0, 255, 0), -1)  # Green for other landmarks

                # Calculate the Mouth Aspect Ratio (MAR) for yawn detection
                mar = calculate_mar(landmarks)

                # Check if the mouth is open (potential yawn)
                if mar > mar_threshold:
                    cv2.putText(frame, 'Yawning Detected', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    else:
                        elapsed_time = time.time() - yawn_start_time
                        if elapsed_time >= yawn_duration_threshold:
                            play_beep()  # Alarm for yawning
                else:
                    yawn_start_time = None  # Reset when mouth is closed

                # Extract the refined eye region using the landmark indices
                left_eye = landmarks.part(36).x, landmarks.part(36).y, landmarks.part(39).x, landmarks.part(39).y
                right_eye = landmarks.part(42).x, landmarks.part(42).y, landmarks.part(45).x, landmarks.part(45).y

                # Calculate the eye aspect ratio (EAR)
                left_ear = ((landmarks.part(39).y - landmarks.part(37).y) + (landmarks.part(40).y - landmarks.part(38).y)) \
                           / (2 * (landmarks.part(41).x - landmarks.part(36).x))
                right_ear = ((landmarks.part(45).y - landmarks.part(43).y) + (landmarks.part(46).y - landmarks.part(44).y)) \
                            / (2 * (landmarks.part(47).x - landmarks.part(42).x))

                # Check if the eyes are closed or open based on EAR threshold
                if left_ear < ear_threshold and right_ear < ear_threshold:
                    cv2.putText(frame, 'Eyes Closed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Track the time when eyes are first detected as closed
                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = time.time()
                    else:
                        # Check if the duration of eyes being closed exceeds the threshold
                        elapsed_time = time.time() - eyes_closed_start_time
                        if elapsed_time >= eyes_closed_duration_threshold:
                            play_beep()  # Play beep sound when eyes are closed for too long
                else:
                    cv2.putText(frame, 'Eyes Open', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # Reset the eyes closed start time when eyes are open
                    eyes_closed_start_time = None

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
                    play_beep()  # Play beep sound when distracted

        # Display the frame
        cv2.imshow('Driver Face Monitoring', frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Process interrupted by user")

finally:
    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()
