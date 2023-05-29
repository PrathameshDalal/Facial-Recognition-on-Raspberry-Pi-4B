import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

# Load the pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
raw_capture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(2)

# Capture and save your face as a reference image
# (adjust the file path and name as needed)
face_reference = cv2.imread('your_face_image.jpg', cv2.IMREAD_GRAYSCALE)

# Start capturing frames from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
    # Convert the raw frame to grayscale for face detection
    gray = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame.array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Compare captured faces with your reference face
        # (uncomment if using a reference face)
        similarity_score = cv2.matchTemplate(gray[y:y+h, x:x+w], face_reference, cv2.TM_CCOEFF_NORMED)
        if similarity_score >= 0.8:
             cv2.putText(frame.array, "Your Name", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow("Face Recognition", frame.array)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)

# Release the camera and close all windows
camera.close()
cv2.destroyAllWindows()
