import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

import numpy as np
import cv2
import time

# Create a MediaPipe HandLandmarker detector. 
# Requires MediaPipe 0.9.1 and above.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.3, 
    min_tracking_confidence=0.3)

def predict(frame):
    """
    ---------------------------------------
    TODO: Task 1.
    Implement the hand landmark prediction.
    ---------------------------------------
    """

    # Perform hand landmark detection
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_result = hands.process(frame)

    return detection_result 

def draw_landmarks_on_image(image, detection_result):
    """
    A helper function to draw the detected 2D landmarks on an image 
    """
    if not detection_result:
        return image 
    
    # hand_landmarks_list = detection_result.hand_landmarks
    # Loop through the detected hands and draw directly on the image
    # Check if detection_result is None or if no hand landmarks were detected
    if detection_result is None or not detection_result.multi_hand_landmarks:
        return image
    
    # Since detection_result.multi_hand_landmarks is not None, proceed to draw landmarks
    for hand_landmarks in detection_result.multi_hand_landmarks:
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks.landmark
        ])
        solutions.drawing_utils.draw_landmarks(
            image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())
    return image

def get_camera_matrix(frame_width, frame_height, scale=1.0): 
    """
    The camera matrix is a matrix of size 3x3 that captures the intrinsic properties of the camera including focal length and center of projection. 
    One can project a 3D point in the camera space to the image plane by multiplying it with the intrinsic matrix. 
    
    For example, let the 3D point by P = np.array([X, Y, Z]) (column vector). Let camera matrix be K. In numpy's code, the projected point is: 
    
    p = K @ P 
    p[0] /= p[2]
    p[1] /= p[2]
    
    Here the division by p[2] is the perspective division. After division, p[0] and p[1] are the x and y coordinate of the image pixel. 
    """
    
    # As we do not know exactly the focal length, we estimate it by a scale of the image size. We can do camera calibration to find a more accurate focal length value but this is out of the scope of this assignment. 
    focal_length = frame_width * scale 
    
    # Note this aspect ratio reflects ratio in the physical pixel size, almost 1, not the aspect ratio between image width and height as in OpenGL. 
    aspect_ratio = 1.0
    
    # Center of projection. We simply take the image center.
    center = (frame_width / 2.0, frame_height / 2.0)
    
    # 3x3 intrinsic matrix
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype = "double"
    )
    return camera_matrix

def get_fov_y(camera_matrix, frame_height):
    """
    Compute the vertical field of view from focal length for OpenGL rendering
    """
    focal_length_y = camera_matrix[1][1]
    fov_y = np.rad2deg(2 * np.arctan2(frame_height, 2 * focal_length_y))
    return fov_y

def get_matrix44(rvec, tvec):
	"""
	Convert the rotation vector and translation vector to a 4x4 matrix
	"""
	rvec = np.asarray(rvec)
	tvec = np.asarray(tvec)
	T = np.eye(4)
	R, jac = cv2.Rodrigues(rvec)
	T[:3, :3] = R
	T[:3, 3] = tvec
	return T

def solvepnp(model_landmarks_list, image_landmarks_list, 
            camera_matrix, frame_width, frame_height): 
    """
    Solve a global rotation and translation to bring the hand model points into the camera space, so that their projected points match the hands. 
    
    Input: 
      model_landmarks_list: a list of 21x3 matrixes representing hand landmarks. The coordinates are relative to the hand center.
      
      image_landmarks_list: a list of 21x2 matrixes representing hand landmarks in image space, normalized to [0, 1]. 
      
    Output: 
      world_landmarks_list: a list of 21x3 matrixes representing hand landmarks in absolute world space.
    """
    if not model_landmarks_list:
        return []

    world_landmarks_list = []
    
    for (model_landmarks, image_landmarks) in zip(model_landmarks_list, image_landmarks_list):
        
        # N x 3 matrix
        model_points = np.float32([[l.x, l.y, l.z] for l in model_landmarks])
        image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in image_landmarks])
        
        world_points = np.copy(model_points)
        
        """
        ----------------------------------------------------------------------
        TODO: Task 2. 
        Call OpenCV's solvePnP function here.
        ----------------------------------------------------------------------
        """
        
        # Task 2: SolvePnP implementation
        ret, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, None)
        
        # Compute the world landmarks using solvePnP
        if ret:
            world_points = get_matrix44(rvec, tvec)
        
        world_landmarks_list.append(world_points)

        # Store all 3D landmarks
        world_landmarks_list.append(world_points)
    
    return world_landmarks_list

def reproject(world_landmarks_list, image_landmarks_list, 
              camera_matrix, frame_width, frame_height): 
    """
    Perform a perspective projection of 3D points onto the image plane
    and return the projected points.
    """
    reprojection_points_list = []
    reprojection_error = 0.0
    for (world_landmarks, image_landmarks) in zip(world_landmarks_list, image_landmarks_list):
        # Perspective projection by multiplying with the intrinsic matrix
        output = world_landmarks.dot(camera_matrix.T)
        
        # Perspective division
        output[:, 0] /= output[:, 2]
        output[:, 1] /= output[:, 2]
        
        # Store the results into a list for visualization later
        reprojection_points_list.append(output[:, :2])
    
        # Calculate the reprojection error, per point
        image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in image_landmarks])
        reprojection_error += np.linalg.norm(output[:, :2] - image_points) / len(output) / len(world_landmarks_list)
    
    return reprojection_error, reprojection_points_list

# # Function added to normalize landmarks
# def normalize_landmarks(model_landmarks, image_width, image_height):
#     # Normalize the landmarks to 2D coordinates (x, y in image space)
#     normalized_landmarks = []

#     for landmark in model_landmarks:
#         # Assuming `landmark` is a 3D point [x, y, z]
#         x, y, z = landmark

#         # Normalize the coordinates to [0, 1] relative to image width and height
#         normalized_x = x / image_width
#         normalized_y = y / image_height

#         # We ignore the z-coordinate in this case
#         normalized_landmarks.append([normalized_x, normalized_y])

#     return np.array(normalized_landmarks)

"""
This is an example main function that displays the video camera and the detection results in 2D landmarks with an OpenCV window.
"""
if __name__ == '__main__':
    # (0) in VideoCapture is used to connect to your computer's default camera
    capture = cv2.VideoCapture(0)
    
    # Initializing current time and precious time for calculating the FPS
    previousTime = 0
    currentTime = 0
    
    while capture.isOpened():
        # capture frame by frame
        ret, frame = capture.read()
    
        # resizing the frame for better view
        aspect_ratio = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (int(720 * aspect_ratio), 720))
        # frame = cv2.flip(frame, 1)

        # Converting the from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        # Making predictions
        detection_result = predict(frame)
    
        # Visualize 2D landmarks
        frame = draw_landmarks_on_image(frame, detection_result)
        
        """
        -------------------------------------------------------------------
        TODO: Task 2. 
        SolvePnP, and visualize the reprojected landmarks. 
        The reprojected points should be close enought to the 2D landmarks
        -------------------------------------------------------------------
        """

        # camera_matrix = get_camera_matrix(frame.shape[1], frame.shape[0], scale=1.0)

        # world_landmarks_list = []
        # reprojection_points_list = []
        # # world_landmarks_list = solvepnp(...)
        
        # # Defining 3D hand landmarks relative to the hand's center (e.g., palm center)
        # model_landmarks_list  = [
        # [0.0, 0.0, 0.0],  # 0 - Palm center
        # [0.0, -0.050, 0.0],  # 1 - Thumb base
        # [0.035, -0.100, 0.0],  # 2 - Thumb first joint
        # [0.070, -0.150, 0.0],  # 3 - Thumb second joint
        # [0.105, -0.200, 0.0],  # 4 - Thumb tip
        # [0.0, 0.050, 0.0],  # 5 - Index base
        # [0.045, 0.100, 0.0],  # 6 - Index first joint
        # [0.090, 0.150, 0.0],  # 7 - Index second joint
        # [0.135, 0.200, 0.0],  # 8 - Index tip
        # [0.0, 0.150, 0.0],  # 9 - Middle base
        # [0.045, 0.200, 0.0],  # 10 - Middle first joint
        # [0.090, 0.250, 0.0],  # 11 - Middle second joint
        # [0.135, 0.300, 0.0],  # 12 - Middle tip
        # [0.0, 0.250, 0.0],  # 13 - Ring base
        # [-0.045, 0.300, 0.0],  # 14 - Ring first joint
        # [-0.090, 0.350, 0.0],  # 15 - Ring second joint
        # [-0.135, 0.400, 0.0],  # 16 - Ring tip
        # [0.0, 0.350, 0.0],  # 17 - Pinky base
        # [-0.045, 0.400, 0.0],  # 18 - Pinky first joint
        # [-0.090, 0.450, 0.0],  # 19 - Pinky second joint
        # [-0.135, 0.500, 0.0],  # 20 - Pinky tip
        # ]

        # image_landmarks_list = normalize_landmarks(detection_result.hand_landmarks, frame.shape[1], frame.shape[0])  # 2D image landmarks

        # # # Call solvePnP
        # world_landmarks_list = solvepnp(model_landmarks_list, image_landmarks_list, camera_matrix, frame.shape[1], frame.shape[0])

        # # reprojection_error, reprojection_points_list = reproject(...)
        
        # reprojection_error, reprojection_points_list = reproject(world_landmarks_list, reprojection_points_list, camera_matrix, frame.shape[1], frame.shape[0])


        # for hand_landmarks in reprojection_points_list:
        #     for l in hand_landmarks:
        #         cv2.circle(frame, (int(l[0]), int(l[1])), 3, (0, 0, 255), 2)
        
        # Calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        
        # Displaying FPS on the image
        cv2.putText(frame, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
        # Display the resulting image
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("", frame)
    
        # Enter key 'q' to break the loop
        if cv2.waitKey(5) & 0xFF == 27:
            break
    
    # When all the process is done
    # Release the capture and destroy all windows
    capture.release()
    cv2.destroyAllWindows()
