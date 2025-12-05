    ## Handles dealing with cameras and getting real world coordinates of ball with positive y being downwards
    ## pip install opencv-contrib-python==4.12.0.88

import time
import cv2
import numpy as np
from pypylon import pylon
import asyncio
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
yellowLowMask = (20, 110, 100)
sensitiveYellowLowMask  = (15, 120, 120)
yellowHighMask = (35, 255, 255)
########################################

cam_matrix = None
distCoeff = np.zeros((5,1))
camera = None
point = None
frame = None
out = None
half_size = 400
small_half_size = 200
size = None
cx = None
cy = None
u = None
v = None
latest_ball = None
lost_counter = 0
FAST_TRACK = False
LOST_THRESHOLD = 5
width = None
height = None
global_u = None
global_v = None

async def TrackerInitialize():
    global point
    global frame
    global distCoeff
    global cam_matrix
    global camera
    global out
    global half_size
    global size
    global cx
    global cy
    global u
    global v
    global width
    global height
    global globalRadius

    print("Tracker Started")
    # Get the cameras
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    camera.Width.SetValue(camera.Width.Max)   # your desired width (must be ≤ Max)
    camera.Height.SetValue(camera.Height.Max)
    camera.PixelFormat.SetValue("RGB8")
    camera.ExposureAuto.SetValue("Off")
    camera.GainAuto.SetValue("Off")
    camera.BalanceWhiteAuto.SetValue("Continuous")
    camera.ExposureTime.SetValue(5000)
    camera.Gain.SetValue(17)
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    half_frame = cv2.resize(grab_result.Array, (0,0), fx=0.5, fy=0.5)
    frame = cv2.rotate(half_frame, cv2.ROTATE_90_CLOCKWISE)
    size = frame.shape[:2]
    width = size[1]
    height = size[0]
    cx = size[1] // 2
    cy = size[0] // 2
    x1 = cx - half_size
    x2 = cx + half_size
    y1 = cy - half_size
    y2 = cy + half_size
    frame = frame[y1:y2, x1:x2]

    print("Initialized")

    # Initialize VideoWriter with frame size
    # height, width = frame.shape[:2]
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MP4V' for mp4
    # out = cv2.VideoWriter('tracker_output2.avi', fourcc, 20.0, (width, height))

async def track_loop(ballColor):
    global latest_ball
    global camera, cx, cy, half_size, small_half_size, lost_counter, FAST_TRACK, LOST_THRESHOLD, global_u, global_v, globalRadius

    # These store the last crop offsets for coordinate correction
    crop_x = cx - half_size
    crop_y = cy - half_size

    while camera.IsGrabbing():
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        full = grab_result.Array[:, :, ::-1]    # full frame (RGB)

        # 1) Resize + rotate entire frame (DON'T CROP YET)
        resized = cv2.resize(full, (0, 0), fx=0.5, fy=0.5)
        rotated = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)

        if latest_ball is None:
            # Use initial center crop
            x1 = cx - half_size
            x2 = cx + half_size
            y1 = cy - half_size
            y2 = cy + half_size

        else:
            # Convert ball coordinate back to full rotated image
            ball_u = latest_ball[0] + crop_x
            ball_v = latest_ball[1] + crop_y

            # Recompute new crop based on ball location
            x1 = int(ball_u - small_half_size)
            x2 = int(ball_u + small_half_size)
            y1 = int(ball_v - small_half_size)
            y2 = int(ball_v + small_half_size)

        # Save offsets for next loop
        crop_x, crop_y = x1, y1

        # Clamp inside frame
        h, w = rotated.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        # Perform crop
        frame = rotated[y1:y2, x1:x2]

        # Detect ball in new cropped frame
        if FAST_TRACK:
            circles = GetLocationFast(frame,ballColor)
        else:
            circles = GetLocation(frame, ballColor)

        if circles is not None:
            lost_counter = 0
            latest_ball = circles[0][0][:2]  # coords inside the CROP
            globalRadius = circles[0][0][2]
            global_u = latest_ball[0] +crop_x
            global_v = latest_ball[1] +crop_y
            DrawCircles(frame, circles, (255, 0, 0))
        else:
            lost_counter += 1
            latest_ball = None
            global_u = None
            global_v = None
            globalRadius = None

            if lost_counter >= LOST_THRESHOLD:
                # ball disappeared — go back to WIDE mode
                FAST_TRACK = False
                latest_ball = None

        # if frame is not None:
        #     cv2.imshow("Tracker", frame)
            # frame_out = cv2.resize(frame, (width, height))
        # out.write(frame_out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        grab_result.Release()

async def TrackerClose():
    global camera
    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
    print("Tracker Ended")

async def getCameraSize():
    global width,height
    while(height is None):
        await asyncio.sleep(0.1)
    return height,width

def GetLocationFast(frame, color):
    # Much lighter blur
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)

    # HSV conversion
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Select color mask
    if color == 'r':
        mask = cv2.inRange(hsv, redLowMask, redHighMask)
    elif color == 'o':
        mask = cv2.inRange(hsv, orangeLowMask, orangeHighMask)
    elif color == 'b':
        mask = cv2.inRange(hsv, blueLowMask, blueHighMask)
    elif color == 'g':
        mask = cv2.inRange(hsv, greenLowMask, greenHighMask)
    elif color == 'y':
        mask = cv2.inRange(hsv, sensitiveYellowLowMask, yellowHighMask)
    else:
        return None

    # Minimal clean-up to remove noise (fast)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    # 6) Pick largest blob (the ball)
    cnt = max(contours, key=cv2.contourArea)
    # Compute center + radius of minimum enclosing circle
    (x, y), radius = cv2.minEnclosingCircle(cnt)

    if radius < 3:   # ignore tiny noise
        return None

    return int(x), int(y), int(radius)

    # # Gray version for Hough
    # gray = cv2.cvtColor(cv2.bitwise_and(blurred, blurred, mask=mask), cv2.COLOR_BGR2GRAY)

    # # FAST Hough detection tuned for *small ROI*
    # circles = cv2.HoughCircles(
    #     gray,
    #     cv2.HOUGH_GRADIENT,
    #     dp=1.0,
    #     minDist=40,            # small crop → smaller distance
    #     param1=80,             # lower edge threshold
    #     param2=15,             # lower center threshold (more sensitive)
    #     minRadius=1,
    #     maxRadius=40
    # )

    # return circles

def GetLocation(frame, color):
    # Uncomment for gaussian blur
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
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
    circles = cv2.HoughCircles(result, cv2.HOUGH_GRADIENT, 1.2, 300, param1=100, param2=20, minRadius=5, maxRadius=50)
    return circles

def DrawCircles(frame, circles, dotColor):
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle corresponding to the center of the circle
            # The circles and rectangles are drawn on the original image.
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), dotColor, -1)


def Calibrate(num_frames=15):
    global point
    global frame
    global distCoeff
    global cam_matrix
    size = (9,6)
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

def run_async_in_thread(coro):
    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro)
        loop.close()

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return t

async def main():
    global u, v
    await TrackerInitialize()
    # Calibrate()
    next = input("stop program?")
    run_async_in_thread(track_loop('y'))
    while (next != "y"):
        next = input("stop program?")
    await TrackerClose()

if __name__ == '__main__':
    asyncio.run(main())
