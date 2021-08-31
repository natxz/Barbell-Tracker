import cv2
# import time
import numpy as np
import imutils
# import math
from application.body_functions.show_heatmap import show_heatmap as heatmap
from application.body_functions.functions import getAngleC, getDistance, getMidPoint, plotPoint, print_pose_elements
from application.body_functions.squat.check_points import get_point_estimations
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt

score = 0
protoFile = "../../pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "../../pose/mpi/pose_iter_160000.caffemodel"
nPoints = 15
POSE_NAMES = {
    "HEAD": 0,
    "NECK": 1,
    "RSHOULDER": 2,
    "RELBOW": 3,
    "RHAND": 4,
    "LSHOULDER": 5,
    "LELBOW": 6,
    "LHAND": 7,
    "RHIP": 8,
    "RKNEE": 9,
    "RANKLE": 10,
    "LHIP": 11,
    "LKNEE": 12,
    "LANKLE": 13,
    "CHEST": 14
    }

POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7],
              [1, 14], [-1, 8], [8, 9], [9, 10], [-1, 11], [11, 12], [12, 13], [14, -1]]
input = input("Enter IMAGE:")
img = cv2.imread(f"{input}.jpg")
centres = []
# img = cv2.imread("../deadlift_bad.jpeg")
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)
bar_distance_threshold = 55
back_angle_threshold = 10
imgcopy = np.copy(img)
imgWidth = img.shape[1]
imgHeight = img.shape[0]
probability_threshold = 0.1
angles_knee = []
angles_hip = []

# read the network
network = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

w = 368
h = 368
inputBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (w, h), (0, 0, 0), swapRB=False, crop=False)
network.setInput(inputBlob)
output = network.forward()
H = output.shape[2]
W = output.shape[3]
print(output.shape)
blurred = cv2.GaussianBlur(imgcopy, (11, 11), 0)
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, greenLower, greenUpper)
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
center = None
if len(cnts) > 0:

    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    if radius > 10:

        cv2.circle(imgcopy, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(imgcopy, center, 5, (0, 0, 255), -1)
        centres.append(center)

# Generate Heatmap to ensure we are getting all of the points
i = 10
probability_map = output[0, i, :, :]
heatmap(probability_map, img)

# Place to store the points detected
pts = []
for i in range(15):
    probability_map = output[0, i, :, :]
    minimum, probability, minlocation, point = cv2.minMaxLoc(probability_map)
    x = (imgWidth * point[0]) / W
    y = (imgHeight * point[1]) / H

    if probability > probability_threshold:
        pts.append((int(x), int(y)))
        # print(POSE_NAMES[i], (int(x), int(y)))
    else:
        pts.append(None)
Pn = {
    "HEAD": pts[0],
    "NECK": pts[1],
    "RSHOULDER": pts[2],
    "RELBOW": pts[3],
    "RHAND": pts[4],
    "LSHOULDER": pts[5],
    "LELBOW": pts[6],
    "LHAND": pts[7],
    "RHIP": pts[8],
    "RKNEE": pts[9],
    "RANKLE": pts[10],
    "LHIP": pts[11],
    "LKNEE": pts[12],
    "LANKLE": pts[13],
    "CHEST": pts[14]
    }
# print_pose_elements(Pn)
try:
    head_length = getDistance(pts[POSE_NAMES["HEAD"]], pts[POSE_NAMES["NECK"]])
except Exception:
    head_length = None
if head_length:
    foot_size = head_length * .85
else:
    forearm_right = getDistance(pts[POSE_NAMES["RHAND"]], pts[POSE_NAMES["RELBOW"]])
    forearm_left = getDistance(pts[POSE_NAMES["LHAND"]], pts[POSE_NAMES["LELBOW"]])
    foot_size = (forearm_right + forearm_left) / 2
    head_length = (foot_size / 85) * 100
pts = get_point_estimations(pts, POSE_NAMES, head_length)
print_pose_elements(Pn)
print(center)
if None not in (pts[POSE_NAMES["RANKLE"]], pts[POSE_NAMES["LANKLE"]]):
    # ============================================================================================
    # SCORE CHECK 1
    # Check that the par is positioned over mid foot
    # ============================================================================================

    ankle_point = getMidPoint(pts, POSE_NAMES, "LANKLE", "RANKLE")
    pts.append(ankle_point)
    plotPoint(imgcopy, ankle_point, 18)
    bar_x_distance_ankles = abs(center[0] - ankle_point[0])
    print("BAR X DISTANCE FROM ANKLES: ", bar_x_distance_ankles)
    if (foot_size / 2) + bar_distance_threshold >= bar_x_distance_ankles >= (foot_size / 2) - bar_distance_threshold:
        score += 1
        print("+1 Score for bar position", bar_x_distance_ankles, foot_size)
    else:
        print("No score for bar position", bar_x_distance_ankles, foot_size)
    # print(bar_x_distance_ankles, score, ankle_point[0], foot_size / 2)

    # ============================================================================================
    # SCORE CHECK 2
    # Check that the athlete's back is straight
    # ============================================================================================
    hips_point = getMidPoint(pts, POSE_NAMES, "LHIP", "RHIP")
    pts.append(hips_point)
    plotPoint(imgcopy, hips_point, 19)

    shoulders_point = center
    pts.append(shoulders_point)
    plotPoint(imgcopy, shoulders_point, 20)

    plotPoint(imgcopy, pts[POSE_NAMES["CHEST"]], 21)
    if pts[POSE_NAMES["HEAD"]] and pts[POSE_NAMES["NECK"]]:
        hl2 = getDistance(pts[POSE_NAMES["HEAD"]], shoulders_point)
        hl2 = (hl2 + (getDistance(pts[POSE_NAMES["HEAD"]], pts[POSE_NAMES["NECK"]])))/2
        print("HEADLENGTH1: ", head_length, "HEADLENGTH2: ", hl2)

    back_angle_chest = getAngleC(hips_point, pts[POSE_NAMES["CHEST"]], pts[POSE_NAMES["NECK"]])
    back_angle_shoulder = getAngleC(hips_point, shoulders_point, pts[POSE_NAMES["NECK"]])
    print(back_angle_chest, back_angle_shoulder)

    if max(back_angle_chest, back_angle_shoulder) <= back_angle_threshold:
        score += 1
        print("+1 score for Back angle", max(back_angle_chest, back_angle_shoulder))
    else:
        print("No score for back angle", max(back_angle_chest, back_angle_shoulder))

    knees_point = getMidPoint(pts, POSE_NAMES, "LKNEE", "RKNEE")
    knee_angle = getAngleC(hips_point, ankle_point, knees_point)
    angles_knee.append(knee_angle)

    hip_angle = getAngleC(knees_point, pts[POSE_NAMES["NECK"]], hips_point)
    angles_hip.append(hip_angle)

    print("KNEE ANGLE: ", knee_angle, "HIP ANGLE: ", hip_angle)

    # ============================================================================================
    # SCORE CHECK 3
    # Check that the athlete hits correct depth in the squat
    # ============================================================================================
    if hips_point[1] >= knees_point[1]:
        score += 1
        print("+1 score for Depth", hips_point, knees_point)
    else:
        print("No score for Depth", hips_point, knees_point)

    plotPoint(imgcopy, knees_point, 55)
    # cv2.putText(imgcopy, f"Chest Angle:  {back_angle_chest}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, lineType=cv2.FILLED)
    # cv2.putText(imgcopy, f"Shoulder Angle:  {back_angle_shoulder}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, lineType=cv2.FILLED)
    cv2.putText(imgcopy, f"Score:  {score}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, lineType=cv2.FILLED)
    cv2.imshow('copy', imgcopy)
    cv2.waitKey()
else:
    print("Not Enough points")
