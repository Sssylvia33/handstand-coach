\# Handstand Coach



Handstand Coach is a computer vision application that uses a webcam or phone camera to analyze handstand posture and provide explainable feedback.



The project is also an exercise in building a computer vision system with clear architecture, reproducible data pipelines, automated tests, and documentation.



\## Project status



Week 1 in progress: real-time pose detection and skeleton visualization.



\## Week 1 user story



As a user, I want to start Handstand Coach using my webcam, see live video with a detected pose skeleton, and stop the application when I am finished, so that I can confirm real-time pose detection works.



\## Week 1 acceptance criteria



\- The user can start live pose detection from the command line.

\- The application continuously reads and displays webcam frames.

\- A skeleton is drawn when a person is detected with sufficient confidence.

\- When nobody is detected, the video continues and displays `No pose detected`.

\- The application displays approximate processing FPS.

\- Pressing `q` closes the application.

\- An unavailable camera produces a useful error message.

\- Camera and window resources are released when the application exits.



\## Architecture constraints



\- Camera capture, pose estimation, and visualization are separate components.

\- Ultralytics results are converted into application-owned data types.

\- Camera and model code are separated from posture calculations.

\- Important non-camera behavior has automated tests.

\- Installation and execution are documented.



\## Week 1 out of scope



\- Session recording and storage

\- Handstand posture metrics

\- Coaching feedback

\- Multiple-person tracking

\- Custom pose-model training

\- A dedicated phone application



\## Current prototype



`detect.py` is an exploratory script that runs pose estimation on a hard-coded image. It will be preserved as the baseline before the application is refactored.

