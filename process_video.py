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
        if self.model is None:
            # create yolo instance
            self.model = YOLO("yolov8n-seg.pt")
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
                    #get generated results from yolo
                    annotated_frame = results[0].plot()
                    
                    #save out to a file
                    out.write(annotated_frame)

                    if current_time - prev_time > 1:
                        calc_fps = frames
                        frames = int(0)
                        print(calc_fps)
                        prev_time = current_time

                    if self.__show_result:
                        annotated_frame = cv2.putText(annotated_frame, str(calc_fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
                        #show frame on display
                        cv2.imshow("frame", annotated_frame)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple script for working with yolov8')
    parser.add_argument('--play_video', type=str, default='', help='Path to the file')
    parser.add_argument('--show_frames', type=bool, default=True, help='The user\'s age.')

    args = parser.parse_args()

    obj = RecognizeObjects(args.show_frames)
    if args.play_video != "":
        if os.path.isfile(args.play_video):
            obj.play(args.play_video)
        else:
            print("{} does not exist".format(args.play_video))
    else:
        obj.run()
