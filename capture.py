import cv2
import time
import mediapipe as mp

# Prepare video capture.

currentTime = 0
previousTime = 0

# 0 to take the video frames from the camera device.
# cap = cv2.VideoCapture(0)
# As long as device is ready

video_path = "A&C.mp4"  # Change to your video file path
cap = cv2.VideoCapture(video_path)  # Read from file instead of webcam

# Get video properties for saving output
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the video writer (output file format)
output_file = "Generalisation.mp4"  # Change to .mp4 if needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawingSpec = mp_drawing.DrawingSpec(color=(255, 255, 255),thickness=1,circle_radius=3)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=9,            # to detect multiple faces
    refine_landmarks=True,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.3
)

while cap.isOpened():

    # Read the video frame
    success, image = cap.read()
    if not success:
        continue

    # Feeding frames to mediaPipe for face landmark detector
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawingSpec,
                connection_drawing_spec=drawingSpec)

    # Calculating the FPS
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    # Save the frame to the output file
    out.write(image)

    # Flip the image for a front-facing webcam view
    image = cv2.flip(image, 1)

    # Resizing the frame for display only
    aspect_ratio = image.shape[1]/image.shape[0]
    image = cv2.resize(image, (int(512*aspect_ratio),512))

    # Displaying FPS on the image
    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

    # Display
    cv2.imshow('OpenCV camera', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Clean up after the loop
cap.release()
out.release()
cv2.destroyAllWindows()