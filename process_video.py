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
    __show_result = True
    __classes = None
    __requested_classes = ["cat"]
    conf_thresh = 0.1
    __colors = {}

    def __init__(self, shwo_res):
        self.__show_result = shwo_res
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
        for key in self.__classes:
            # a work around for now to show all detected classes
            self.__requested_classes.append(self.__classes[key])
        self.__colors = {self.__classes[class_name] : random.choices(range(256), k=3) for class_name in self.__classes}
        prev_time = time.time()
        frames = int(0)
        calc_fps = int(0)
        try:
            for path in self.input_files_list:
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
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # Create the VideoWriter object
                out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
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
        for obj in objects["objects"]:
            # loop throuth all detecte objects
            color = self.__colors[obj["name"]]

            # draw shape of object
            cv2.polylines(frame, obj["shape"], True, color, 5)
            
            y_text = obj['y'] - 15 if obj['y'] - 15 > 15 else obj['y'] + 15
            label = label_f.format(obj["name"], obj["confidence"])
            #add object info above shape
            frame = cv2.putText(frame, label, (20, y_shift),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # add object info on left side for statistics
            frame = cv2.putText(frame, label, (obj['x'], y_text),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y_shift += 20

        return frame


    def __process_yolo_outs(self, results):
        ret_value = {'objects': []}
        shape_points = []
        class_ids=[]
        confidences=[]
        boxes=[]
        
        for result in results:
                boxes_data = result.boxes
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
                    if confidence > self.conf_thresh:
                        inst = {"name": obj_name, "confidence":confidence , "x": boxes[idx][0], "y": boxes[idx][1], "w": boxes[idx][2], "h": boxes[idx][3], "shape":shape_points[idx]}
                        ret_value['objects'].append(inst)
        return ret_value
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple script for working with yolov8')
    parser.add_argument('--play', type=str, default='', help='Path to the file')
    parser.add_argument('--show_frames', type=bool, default=True, help='flag to show results on the display during processing')

    args = parser.parse_args()

    obj = RecognizeObjects(args.show_frames)
    if args.play != "":
        if os.path.isfile(args.play):
            obj.play(args.play)
        else:
            print("{} does not exist".format(args.play_video))
    else:
        obj.run()
