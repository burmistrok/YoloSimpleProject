import cv2
from ultralytics import YOLO
import glob
import os
import sys

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

model = YOLO("yolov8n-seg.pt")

def detect_obj_in_frame(frame):
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("frame", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise Exception("Q pressed by the User")


class RecognizeObjects():

    #path were input files are stored
    in_path = ROOT_PATH + "/Demo/"
    #path were outup fill will be stored
    out_path = ROOT_PATH + "/out/"
    # list of files to work with
    input_files_list = []
    # list of out files
    output_file_list = []

    def __init__(self):
        mask = self.in_path + "*.mp4"
        print(mask)
        for path in glob.glob(mask):

            #find all files added for processing
            self.input_files_list.append(path)
            print(path)
            # create paths to all exit files
            path_s = path.split("/")
            name = path_s[-1].split(".")[0]
            out_name = self.out_path + name + ".mp4"
            self.output_file_list.append(out_name)
            print(out_name)

    def play(self, input_files = True):
        self.process_files(
            input_files=input_files,
            function=lambda frame: (cv2.imshow("frame", frame), cv2.waitKey(1))
        )



    def process_files(self, input_files = True, function = None):
        if input_files is not False:
            files_to_play = self.input_files_list
        else:
            files_to_play = self.output_file_list
        for path in files_to_play:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print("Error opening video file:", path)
                return
            ret, frame = cap.read()
            while ret:
                if function is not None:
                    function(frame)
                ret, frame = cap.read()
            cap.release() 


    def detect_objs(self):
        try:
            self.process_files(
                function=detect_obj_in_frame
            )
        except Exception as e:
            print(e)
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    obj = RecognizeObjects().detect_objs()
