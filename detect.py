# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 10:02:50 2026

@author: jiaqi
"""
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

image_path = "pose_2.jpg"
results = model(image_path)

for result in results:
    if result.keypoints is not None:
        keypoints = result.keypoints.xy.cpu().numpy()

        if len(keypoints) > 0:
            person_kp = keypoints[0]

            # 5 left shoulder, 6 right shoulder
            # 9 left wrist, 10 right wrist
            # 11 left hip, 12 right hip

            l_shoulder = person_kp[5]
            l_wrist = person_kp[9]
            l_hip = person_kp[11]

            print("--successfuly extract keypoints--")
            print(f"Shoulder X,Y : {l_shoulder}")
            print(f"Wrist X,Y : {l_wrist}")
            print(f"Hip X,Y : {l_hip}")

            # init theory wrist x is close to hip x, indiates
            # center of gravity is above pivot point
            x_distance = abs(l_hip[0] - l_wrist[0])
            print(
                f"Distance between center of gravity and pivot point (wrist) : {x_distance:.2f} pixel"
            )

            annotated_frame = result.plot()
            scale_percent = 65
            new_width = int(annotated_frame.shape[1] * scale_percent / 100)
            new_height = int(annotated_frame.shape[0] * scale_percent / 100)
            resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

            cv2.imshow("YOLOv8 Pose Detection", resized_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
