from pypylon import pylon
import cv2
import numpy as np

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

while camera.IsGrabbing():
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grab_result.GrabSucceeded():
        img = grab_result.Array
        cv2.imshow("Basler", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Grab failed")

    grab_result.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
