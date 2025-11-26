    ## Handles dealing with cameras and getting real world coordinates of ball with positive y being downwards
    ## pip install opencv-contrib-python==4.12.0.88

import socket
import time
from queue import Queue
import cv2
import numpy as np
import threading

#####HSV Colour Ranges#################
#If the ball is red (0-10) or (170-180)
redLowMask = (160,100,100)
redHighMask = (180, 255, 255)

#If the ball is blue
blueLowMask = (100, 150, 0)
blueHighMask = (140, 255, 255)

#If the ball is orange
orangeLowMask = (5, 50, 50)
orangeHighMask = (20, 255, 255)

#If the ball is green
greenLowMask= (90, 50, 50)
greenHighMask= (150, 255, 255)

# If the ball is yellow
yellowLowMask = (20, 100, 100)
yellowHighMask = (30, 255, 255)
########################################

class Tracker:

    def __init__(self, ballColor):
        self.ready = False

        self.left_cam_extrinsic = np.hstack((np.eye(3), np.zeros((3,1)))) ## cam 1 facing z axis
        self.right_cam_extrinsic = np.array([[1,0,0,0.3],  ## Assuming camera two is negative x and position is (1,0,1)
                                            [0,1,0,0.16],
                                            [0,0,1,0]], dtype=float)
        # self.right_cam_extrinsic = np.array([[np.cos(np.pi/4),0,-np.sin(np.pi/4),0.782],
        #                                      [np.sin(np.pi/4),1,np.cos(np.pi/4),0.17],
        #                                      [0,0,1,0]])

        self.left_cam_matrix = np.eye(3,3)
        self.right_cam_matrix = np.eye(3,3)
        self.distCoeff_left = np.zeros((5,1))
        self.distCoeff_right = np.zeros((5,1))

        self.left_point = 0
        self.right_point = 0

        self.frame_left = 0
        self.frame_right = 0
        thread = threading.Thread(target=self.TrackerThread, args=(ballColor,), daemon=True)
        thread.start()

    def TrackerThread(self, ballColor):
        print("Tracker Started")
        print("hello")
        # Get the cameras
        
        cam_left = cv2.VideoCapture(0)
        cam_right = cv2.VideoCapture(1)

        print("left cam opened:", cam_left.isOpened())
        print("right cam opened:", cam_right.isOpened())

        if cam_left.isOpened() and cam_right.isOpened():
            rval_left, self.frame_left = cam_left.read()
            rval_right, self.frame_right = cam_right.read()
            print("Both cameras found")
        else:
            print("One or Both Cameras not found")
            return
        self.ready == True


        # background_left = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        # background_right = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
        while rval_left and rval_right:
            # Handle current frame for left camera
            rval_left, self.frame_left = cam_left.read()
            frame_left = self.frame_left
            left_circle = self.GetLocation(frame_left, ballColor)
            # left_circle = self.GetLocation(frame_left, ballColor, background_left)
            self.DrawCircles(frame_left, left_circle, (255, 0, 0))

            # Handle current frame for right camera
            rval_right, self.frame_right = cam_right.read()
            frame_right = self.frame_right
            right_circle = self.GetLocation(frame_right, ballColor)
            # right_circle = self.GetLocation(frame_right, ballColor, background_right)
            self.DrawCircles(frame_right, right_circle, (0, 0, 255))

            if left_circle is not None:
                self.left_point = left_circle[0]

            if right_circle is not None:
                self.right_point = right_circle[0]

            # Shows the original image with the detected circles drawn.
            combined = cv2.hconcat([frame_left, frame_right])
            cv2.imshow("Left | Right", combined)

            # check if esc key pressed
            key = cv2.waitKey(20)
            if key == 27:
                break

        cam_left.release()
        cv2.destroyAllWindows()
        print("Tracker Ended")

    def GetLocation_bis(self, frame, background):
        blurred = cv2.GaussianBlur(frame, (10,10), 0)
        mask = background.apply(blurred)
        # Perform erosion and dilation in the image (in 5x5 pixels squares) in order to reduce the "blips" on the mask
        mask = cv2.erode(mask, np.ones((5, 5),np.uint8), iterations=2)
        mask = cv2.dilate(mask, np.ones((5, 5),np.uint8), iterations=4)
        # masked_blurred = cv2.bitwise_and(frame,frame, mask= mask)
        masked_blurred = cv2.bitwise_and(blurred,blurred, mask= mask)
        # Convert the masked image to gray scale (Required by HoughCircles routine)
        result = cv2.cvtColor(masked_blurred, cv2.COLOR_BGR2GRAY)
        # Detect circles in the image using Canny edge and Hough transform
        circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, 1.2, 300, param1=100, param2=20, minRadius=10, maxRadius=200)
        return circles



    def GetLocation(self, frame, color):
        # Uncomment for gaussian blur
        #blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        blurred = cv2.medianBlur(frame,11)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        if color == 'r':
            # Red Tracking
            mask = cv2.inRange(hsv, redLowMask, redHighMask)
        if color == 'o':
            # Orange Tracking
            mask = cv2.inRange(hsv, orangeLowMask, orangeHighMask)
        if color == 'b':
            # Blue Tracking
            mask = cv2.inRange(hsv, blueLowMask, blueHighMask)
        if color == 'g':
            # Green Tracking
            mask = cv2.inRange(hsv, greenLowMask, greenHighMask)
        if color == 'y':
            mask = cv2.inRange(hsv, yellowLowMask, yellowHighMask)
        # Perform erosion and dilation in the image (in 11x11 pixels squares) in order to reduce the "blips" on the mask
        mask = cv2.erode(mask, np.ones((5, 5),np.uint8), iterations=2) ## change if hard to detect ball
        mask = cv2.dilate(mask, np.ones((5, 5),np.uint8), iterations=5)
        # Mask the blurred image so that we only consider the areas with the desired colour
        masked_blurred = cv2.bitwise_and(blurred,blurred, mask= mask)
        # masked_blurred = cv2.bitwise_and(frame,frame, mask= mask)
        # Convert the masked image to gray scale (Required by HoughCircles routine)
        result = cv2.cvtColor(masked_blurred, cv2.COLOR_BGR2GRAY)
        # Detect circles in the image using Canny edge and Hough transform
        circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, 1.5, 300, param1=100, param2=20, minRadius=10, maxRadius=200)
        return circles

    def DrawCircles(self, frame, circles, dotColor):
        # ensure at least some circles were found
        if circles is not None:
            # convert the (x, y) coordinates and radius of the circles to integers
            circles = np.round(circles[0, :]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x, y, r) in circles:
                #print("Circle: " + "("+str(x)+","+str(y)+")")
                # draw the circle in the output image, then draw a rectangle corresponding to the center of the circle
                # The circles and rectangles are drawn on the original image.
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), dotColor, -1)


    def Calibrate_Accurate(self, num_frames=15):
        """
        Calibrates left and right cameras using ChArUco board.
        Uses the live frames from TrackerThread: self.frame_left and self.frame_right.
        Collects num_frames valid frame pairs where ChArUco corners are detected.
        """

        # --- ArUco dictionary and Charuco board ---
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)

        board = cv2.aruco.CharucoBoard(
            size=(5, 7),
            squareLength=0.04, # m
            markerLength=0.02, # m
            dictionary=dictionary
        )

        # Storage for valid corners/ids
        left_corners_list = []
        left_ids_list = []
        right_corners_list = []
        right_ids_list = []
        img_size = None

        print("Collecting frame pairs for calibration...")

        while len(left_corners_list) < num_frames:
            # Grab the latest frames from TrackerThread
            input("Press Enter to sample a frame (", len(left_corners_list)+1, "out of", num_frames, "): ")
            frame_left = self.frame_left
            frame_right = self.frame_right

            if frame_left is None or frame_right is None:
                time.sleep(0.1)
                continue

            # --- Process LEFT camera ---
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            corners_left, ids_left, _ = detector.detectMarkers(gray_left)
            if ids_left is not None and len(ids_left) > 0:
                charuco_corners_left, charuco_ids_left = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners_left,
                    markerIds=ids_left,
                    image=gray_left,
                    board=board
                )
                if charuco_ids_left is not None and len(charuco_ids_left) > 3:
                    left_corners_list.append(charuco_corners_left)
                    left_ids_list.append(charuco_ids_left)
                    img_size = gray_left.shape[::-1]

            # --- Process RIGHT camera ---
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            corners_right, ids_right, _ = detector.detectMarkers(gray_right)
            if ids_right is not None and len(ids_right) > 0:
                charuco_corners_right, charuco_ids_right = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners_right,
                    markerIds=ids_right,
                    image=gray_right,
                    board=board
                )
                if charuco_ids_right is not None and len(charuco_ids_right) > 3:
                    right_corners_list.append(charuco_corners_right)
                    right_ids_list.append(charuco_ids_right)

        # --- Calibrate LEFT camera ---
        retVal_left, self.left_cam_matrix, self.distCoeff_left, distCoeffs_left, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=left_corners_list,
            charucoIds=left_ids_list,
            board=board,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None
        )

        # --- Calibrate RIGHT camera ---
        retVal_right, self.right_cam_matrix, self.distCoeff_right, distCoeffs_right, _ = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=right_corners_list,
            charucoIds=right_ids_list,
            board=board,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None
        )

        print("Calibration complete!")
        print("Left camera matrix:", self.left_cam_matrix)
        print("Left camera distortion coeffecients:", self.distCoeff_left)
        print("Right camera matrix:", self.right_cam_matrix)
        print("Right camera distortion coeffecients:", self.distCoeff_right)


    def Calibrate(self, num_frames=15):
        size = (8,6)
        square_size = 0.024 # m

        left_corners_list = []
        left_obj_list = []
        right_corners_list = []
        right_obj_list = []
        img_size = None
        objp = np.zeros((size[0] * size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
        objp *= square_size

        while len(left_corners_list) < num_frames:
            in_string = "Press Enter to sample a frame ("+ str(len(left_corners_list)+1)+ " out of "+ str(num_frames) + "): "
            input(in_string)
            frame_left = self.frame_left
            frame_right = self.frame_right

            if frame_left is None or frame_right is None:
                num_frames -= 1
                time.sleep(0.1)
                continue

         # --- Process LEFT camera ---
            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            bool_left, corners_left = cv2.findChessboardCorners(gray_left, size,None)
            if bool_left:
                left_corners_list.append(corners_left)
                left_obj_list.append(objp.copy())
            else:
                print("Left camera could not find corners")


            # --- Process RIGHT camera ---
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            bool_right, corners_right = cv2.findChessboardCorners(gray_right, size,None)
            if bool_right:
                right_corners_list.append(corners_right)
                right_obj_list.append(objp.copy())
            else:
                print("Right camera could not find corners")

        retVal, self.left_cam_matrix, self.distCoeff_left, _, _ = cv2.calibrateCamera(left_obj_list, 
                        left_corners_list, gray_left.shape[::-1], None, None)
        retVal, self.right_cam_matrix, self.distCoeff_right, _, _ = cv2.calibrateCamera(right_obj_list, 
                        right_corners_list, gray_right.shape[::-1], None, None)
        print("Calibration complete!")
        print("Left camera matrix:", self.left_cam_matrix)
        print("Left camera distortion coeffecients:", self.distCoeff_left)
        print("Right camera matrix:", self.right_cam_matrix)
        print("Right camera distortion coeffecients:", self.distCoeff_right)
        

        
    def getCircleRadius(self):
        return (self.left_point[0][2] + self.right_point[0][2]) /2 ## double check if this works

    def getAccurateBallPosition(self):

        if(self.left_point is None or self.right_point is None):
            return None

        ## Get Points
        u_left,v_left = self.left_point[0][:2]
        u_right,v_right = self.right_point[0][:2]
        point_left = np.array([[u_left, v_left]])
        point_right = np.array([[u_right, v_right]])

        # Undistort
        undistorted_left = cv2.undistortPoints(point_left, self.left_cam_matrix, self.distCoeff_left)
        undistorted_left = undistorted_left.reshape(2,1)

        undistorted_right = cv2.undistortPoints(point_right,self.right_cam_matrix, self.distCoeff_right)
        undistorted_right = undistorted_right.reshape(2,1)

        # Triangulate
        point4D = cv2.triangulatePoints(self.left_cam_extrinsic, self.right_cam_extrinsic,
                                        undistorted_left, undistorted_right)
        
        if (point4D[3] == 0):
            print("Triangulation Failed :(")
        point3D = point4D[:3] / point4D[3] ## Homogenous -> 3D

        print (point3D)
        return point3D
    
if __name__ == '__main__':
    t = Tracker('b')
    input("Press enter to Calibrate")
    t.Calibrate()
    while 1:
        input("enter to getLocation")
        t.getAccurateBallPosition()
        if (input("end?") == "y"):
            break
    

        # ## deprecated

        # def getBallPosition(self):
        #     x,y = self.left_point[0][:2]
        #     left_pos = np.array([x,y])
        #     x,y = self.right_point[0][:2]
        #     right_pos = np.array([x,y])
        #     xL = left_pos[0] * self.left_cam_pixels_per_cm
        #     xR = right_pos[0] * self.right_cam_pixels_per_cm
        #     denom = xL * self.right_focal_length - xR * self.left_focal_length
        #     z = (self.left_focal_length*self.right_focal_length*self.distance_bw_cameras) / denom
        #     x = (xL * self.right_focal_length * self.distance_bw_cameras) / denom
        #     y = (z * left_pos[1])/ self.left_focal_length
        #     return np.array([x,y,z])