import cv2
try:
    from ultralytics import YOLO
except Exception as e:
    print(e)
import glob
import os
import sys
import argparse
import time
import random
import numpy as np

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
white = (255, 255, 255)
black = (0, 0, 0)
yellow = (0, 255, 225)

class RecognizeObjects():

    #path were input files are stored
    in_path = ROOT_PATH + "/Demo/"
    #path were outup fill will be stored
    out_path = ROOT_PATH + "/out/"
    # list of files to work with
    input_files_list = []
    # list of out files
    output_file_list = []
    # yolo model instance
    model = None
    __show_result = False
    __classes = None
    __requested_classes = ["cat", "bird", "mouse"]
    conf_thresh = 0.1
    __colors = {}
    actual_collor = yellow
    is_allowed = True
    does_pass_thr = False
    pass_thresold = -1
    frame_width = 0
    frame_height = 0

    def __init__(self, shwo_res):
        if shwo_res == "True":
            self.__show_result = True
        mask = self.in_path + "*.mp4"
        print(mask)
        for path in glob.glob(mask):

            #find all files added for processing
            self.input_files_list.append(path)
            print(path)

        ## make sure out path exists
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)


    def play(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("Error opening video file:", path)
            return
        ret, frame = cap.read()
        while ret:
            cv2.imshow("frame", frame)
            ret, frame = cap.read()
            if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
                break
        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        # create yolo instance
        self.model = YOLO("yolov8n-seg.pt")
        self.__classes = self.model.names
        self.__colors = {self.__classes[class_name] : random.choices(range(256), k=3) for class_name in self.__classes}
        prev_time = time.time()
        frames = int(0)
        calc_fps = int(0)
        try:
            for path in self.input_files_list:

                self.actual_collor = yellow
                self.is_allowed = True
                self.does_pass_thr = False
                # create paths to all exit files
                path_s = path.split("/")
                name = path_s[-1].split(".")[0]
                out_path = self.out_path + name + ".mp4"
                # check if the files exists
                if not os.path.isfile(path):
                    print("{} does not exist".format(path))
                # open video file
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    print("Error opening video file:", path)
                    return
                # colect metadata for out file
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.pass_thresold = int(self.frame_height*0.6)
                # Create the VideoWriter object
                out = cv2.VideoWriter(out_path, fourcc, fps, (self.frame_width, self.frame_height))
                ret, frame = cap.read()
                while ret:
                    current_time = time.time()
                    #process frame with yolo      
                    results = self.model(frame)  
                    #parce generated results from yolo
                    results = self.__process_yolo_outs(results)
                    # draw all metadata to image
                    frame = self.__process_frame(results, frame)
                    # save frame to output video file
                    out.write(frame)

                    if current_time - prev_time > 1:
                        # implemetation for fps calculation
                        calc_fps = frames
                        frames = int(0)
                        print(calc_fps)
                        prev_time = current_time

                    if self.__show_result:
                        frame = cv2.putText(frame, str(calc_fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
                        #show frame on display
                        cv2.imshow("frame", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
                            raise Exception("User presed Q button")
                    # get next frame
                    ret, frame = cap.read()
                    frames += 1
                # release resources
                cap.release()
                out.release()
        except Exception as e:
            print(e)
            cap.release()
            out.release()
        if self.__show_result:
            # close window with last frame
            cv2.destroyAllWindows()


    def __process_frame(self, objects, frame):
        y_shift = 70
        label_f = "{}: {:.2f}"
        
        h_thr = self.pass_thresold
        
        if 0 < len(objects["objects"]):
            # loop throuth all detecte objects
            for obj in objects["objects"]:

                # check if pray is detected
                if obj["name"] == "bird" or obj["name"] == "mouse":
                    # bloc door in case a pray is detected
                    self.is_allowed = False
                    self.actual_collor = red
                elif obj["name"] == "cat":
                    if (obj["y"] + obj["h"]) > h_thr:
                        self.does_pass_thr = True
                        if self.is_allowed is not False:
                            #open door in case it is allowed to enter
                            self.actual_collor = green

            # draw shape of object
            cv2.polylines(frame, obj["shape"], True, self.actual_collor, 5)

            y_text = obj['y'] - 15 if obj['y'] - 15 > 15 else obj['y'] + 15
            label = label_f.format(obj["name"], obj["confidence"])
            #add object info above shape
            frame = cv2.putText(frame, label, (obj['x'], y_text),cv2.FONT_HERSHEY_SIMPLEX, 1, self.actual_collor, 2)


        #draw threshold line
        frame = cv2.line(frame, (0, h_thr), (self.frame_width, h_thr), self.actual_collor, 2)
        #put meta data
        frame = cv2.putText(frame, "is allowed: {}".format(self.is_allowed), (50, y_shift),cv2.FONT_HERSHEY_SIMPLEX, 1, self.actual_collor, 2)
        y_shift += 40
        frame = cv2.putText(frame, "does it pass thr: {}".format(self.does_pass_thr), (50, y_shift),cv2.FONT_HERSHEY_SIMPLEX, 1, self.actual_collor, 2)

        return frame


    def __process_yolo_outs(self, results):
        ret_value = {'objects': []}
        shape_points = []
        class_ids=[]
        confidences=[]
        boxes=[]
        
        for result in results:
                boxes_data = result.boxes
                
                if result.masks is not None:
                    for points in result.masks.xy:
                        shape_points.append(np.int32([points]))
                for obj in boxes_data.cls:
                    class_ids.append(self.__classes[int(obj)])
                for obj in boxes_data.conf:
                    confidences.append(float(obj))
                for obj in boxes_data.xywh:
                    x_c = int(obj[0])
                    y_c = int(obj[1])
                    w = int(obj[2])
                    h = int(obj[3])
                    x = int(x_c - int(w/2))
                    if x < 0:
                        x = 0
                    y = int(y_c - int(h/2))
                    if y < 0:
                        y = 0
                    boxes.append([x,y,w,h])
                
                no_of_objs = len(confidences)
                for idx in range(no_of_objs):
                    confidence = float(confidences[idx])
                    obj_name = class_ids[idx]
                    if confidence > self.conf_thresh and obj_name in self.__requested_classes:
                        inst = {"name": obj_name, "confidence":confidence , "x": boxes[idx][0], "y": boxes[idx][1], "w": boxes[idx][2], "h": boxes[idx][3], "shape":shape_points[idx]}
                        ret_value['objects'].append(inst)
        return ret_value
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple script for working with yolov8')
    parser.add_argument('--play', type=str, default='', help='Path to the file')
    parser.add_argument('--show', type=str, default="True", help='flag to show results on the display during processing')

    args = parser.parse_args()

    obj = RecognizeObjects(args.show)
    if args.play != "":
        if os.path.isfile(args.play):
            obj.play(args.play)
        else:
            print("{} does not exist".format(args.play_video))
    else:
        obj.run()

