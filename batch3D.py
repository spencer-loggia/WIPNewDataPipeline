import deeplabcut as dlc
import os
import datetime
import shutil
import sys


class Process3D:

    def __init__(self, file_3d, cal_dir, vid_dir):
        '''
        Function to run a bunch of 3d labellings, while recallibrating 3D net as needed
        and detecting errors.
        :param config_file_3d: path to 3d network config file. This file must be updated
                               to include paths to trained 2D nets and appropriate prefixes.
        :param cal_dir: directory contain folders for each date of processed calibration data.
        :param vid_dir: directory containing folder with snippets for each date.
        '''

        self.vid_dir = str(vid_dir)
        self.cal_dir = cal_dir
        self.videos = os.listdir(vid_dir)
        self.cal = os.listdir(cal_dir)
        self.net = file_3d

    def calibrate(self, cal_date_dir):
        # move cal images to 3D project
        try:
            shutil.rmtree(os.path.join(self.net, 'calibration_images'))
        except FileNotFoundError:
            pass
        shutil.copytree(os.path.join(self.cal_dir, cal_date_dir), os.path.join(self.net, 'calibration_images'))
        net_file = os.path.join(self.net, 'config.yaml')
        dlc.calibrate_cameras(net_file, cbrow=9, cbcol=9, calibrate=True, alpha=0.9)

    def triangulate(self, vid_date_dir):
        net_file = os.path.join(self.net, 'config.yaml')
        try:
            os.mkdir(os.path.join(os.path.split(self.vid_dir)[0], 'predicted_3d'))
        except FileExistsError:
            pass
        dest_path = os.path.join(os.path.split(self.vid_dir)[0], 'predicted_3d', vid_date_dir)
        try:
            os.mkdir(dest_path)
        except FileExistsError:
            pass
        dlc.triangulate(net_file, os.path.join(self.vid_dir, vid_date_dir), filterpredictions=True, destfolder=dest_path)
        dlc.create_labeled_video_3d(config=net_file,
                                    path=[dest_path],
                                    videofolder=os.path.join(self.vid_dir, vid_date_dir))

    def process(self):
        for vid_date in self.videos:
            date = str(self.videos[12:20])
            vdate = datetime.date(month=int(date[0:2]),
                                  day=int(date[2:4]),
                                  year=int(date[4:8]))
            best_cal = None
            best_date = datetime.date(00, 00, 0000)
            for c in self.cal:
                cdate = str(self.cal[5:])
                cdate = datetime.date(month=int(cdate[0:2]),
                                      day=int(cdate[2:4]),
                                      year=int(cdate[4:8]))
                if vdate > cdate > best_date:
                    best_cal = c
                    best_date = cdate
            if best_cal is not None:
                print("Processing " + vid_date + " using cal file from " + str(best_date))
                self.calibrate(best_cal)
                self.triangulate(vid_date)


if __name__ == "__main__":
    net_path = sys.argv[1]
    cal_dir_path = sys.argv[2]
    vid_dir_path = sys.argv[3]
    processor = Process3D(net_path, cal_dir_path, vid_dir_path)
    processor.process()


