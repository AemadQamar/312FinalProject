    ## Handles dealing with cameras and getting real world coordinates of ball with positive y being downwards
    ## pip install opencv-contrib-python==4.12.0.88

import time
import cv2
import numpy as np
from pypylon import pylon
import asyncio

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
yellowHighMask = (35, 255, 255)
########################################

cam_matrix = None
distCoeff = np.zeros((5,1))
camera = None
point = None
frame = None

async def TrackerInitialize():
    global point
    global frame
    global distCoeff
    global cam_matrix
    global camera

    print("Tracker Started")
    # Get the cameras
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    # camera.PixelFormat.SetValue("RGB8")
    # camera.ExposureAuto.SetValue("Off")
    # camera.GainAuto.SetValue("Off")
    # camera.ExposureTime.SetValue(11000)
    # camera.Gain.SetValue(3)
    # camera.BalanceWhiteAuto.SetValue("Continuous")
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grab_result is None:
        print("❌ RetrieveResult returned None")
        return
    if not grab_result.GrabSucceeded():
        print("❌ Grab failed:", grab_result.ErrorCode, grab_result.ErrorDescription)
        return

    img = grab_result.Array
    if img is None or img.size == 0:
        print("❌ Empty frame received")

    frame = cv2.resize(grab_result.Array, (0,0), fx=0.5, fy=0.5)
    print("Initialized")

async def getNextBallInstance(ballColor):
    global point
    global frame
    global distCoeff
    global cam_matrix
    global camera 
    rval = True
    circles = None
    while camera.IsGrabbing() and circles is None:
        # Handle current frame for camera
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        frame = cv2.resize(grab_result.Array[:, :, ::-1], (0,0), fx=0.5, fy=0.5)
        grab_result.Release()
        circles = GetLocation(frame, ballColor)

        DrawCircles(frame, circles, (255, 0, 0))
        cv2.imshow("Tracker", frame)
        key = cv2.waitKey(5)
        if key == 27:
            break

        if circles is None:
            return None
        point = circles[0]
        u,v = point[0][:2]
        # Shows the original image with the detected circle drawn.
        if(cam_matrix is not None):
            np_point = np.array([[u, v]])
            np_point = cv2.undistortPoints(np_point, cam_matrix, distCoeff).reshape(2,1)
            return (np_point[0][0], np_point[1][0])
    return (u,v)

async def TrackerClose():
    global camera
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
    print("Tracker Ended")

def getCameraSize():
    return frame.shape[:2]
def GetLocation(frame, color):
    # Uncomment for gaussian blur
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    # blurred = cv2.medianBlur(frame,11)
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
    mask = cv2.erode(mask, np.ones((3, 3),np.uint8), iterations=2) ## change if hard to detect ball
    mask = cv2.dilate(mask, np.ones((3, 3),np.uint8), iterations=4)
    # Mask the blurred image so that we only consider the areas with the desired colour
    masked_blurred = cv2.bitwise_and(blurred,blurred, mask= mask)
    # masked_blurred = cv2.bitwise_and(frame,frame, mask= mask)
    # Convert the masked image to gray scale (Required by HoughCircles routine)
    result = cv2.cvtColor(masked_blurred, cv2.COLOR_BGR2GRAY)
    # Detect circles in the image using Canny edge and Hough transform
    circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, 1.5, 300, param1=100, param2=20, minRadius=5, maxRadius=200)
    return circles

def DrawCircles(frame, circles, dotColor):
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

def Calibrate_Accurate(num_frames=15):
    global point
    global frame
    global distCoeff
    global cam_matrix

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
    ids_list = []
    img_size = None

    print("Collecting frame pairs for calibration...")

    while len(left_corners_list) < num_frames:
        # Grab the latest frames from TrackerThread
        input("Press Enter to sample a frame (", len(left_corners_list)+1, "out of", num_frames, "): ")
        frame = frame

        if frame is None:
            time.sleep(0.1)
            continue

        # --- Process camera ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=board
            )
            if charuco_ids is not None and len(charuco_ids) > 3:
                left_corners_list.append(charuco_corners)
                ids_list.append(charuco_ids)
                img_size = gray.shape[::-1]

    # --- Calibrate camera ---
    retVal, cam_matrix, distCoeff, _, _ = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=left_corners_list,
        charucoIds=ids_list,
        board=board,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None
    )

    print("Calibration complete!")
    print("Right camera matrix:", cam_matrix)
    print("Right camera distortion coeffecients:", distCoeff)


def Calibrate(num_frames=15):
    global point
    global frame
    global distCoeff
    global cam_matrix
    size = (9,7)
    square_size = 0.027 # m

    corners_list = []
    obj_list = []
    img_size = None
    objp = np.zeros((size[0] * size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    objp *= square_size

    while len(corners_list) < num_frames:
        in_string = "Press Enter to sample a frame ("+ str(len(corners_list)+1)+ " out of "+ str(num_frames) + "): "
        input(in_string)
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        frame = cv2.resize(grab_result.Array[:, :, ::-1], (0,0), fx=0.5, fy=0.5)
        grab_result.Release()

        if frame is None:
            num_frames -= 1
            time.sleep(0.1)
            continue

        # --- Process Camera ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bool, corners = cv2.findChessboardCorners(gray, size,None)
        if bool:
            corners_list.append(corners)
            obj_list.append(objp.copy())
        else:
            print("Camera could not find corners")


    retVal, cam_matrix, distCoeff, _, _ = cv2.calibrateCamera(obj_list,
                    corners_list, gray.shape[::-1], None, None)
    print("Calibration complete!")
    print("Camera matrix:", cam_matrix)
    print("Camera distortion coeffecients:", distCoeff)

async def main():
    await TrackerInitialize()
    Calibrate()
    next = input("stop program?")
    while (next != "y"):
        print(await getNextBallInstance('y'))
    await TrackerClose()

if __name__ == '__main__':
    asyncio.run(main())
    