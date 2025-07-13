# YoloSimpleProject
An implementation for video processing using yolov8.

It open all mp4 fils in Demo folder, find all objects in the clip using yolo8v draw its shapes , add labels with name and confidence above the object and on left side for statistics display output in real time and save it to out folder


Usage:

Put videos in mp4 format in Demo folder and start the script:
python3 process_video.pyy

In case is is neccesary to reduce load, we can disable show results on the displa using "--show False" flag to the command for example:
python3 process_video.py --show False


in case it is necessary to watch out file later, can be used --play flag, for example:
python3 process_video.py --play ./out/video3.mp4


Installation:
pip3 install -r requirements.txt