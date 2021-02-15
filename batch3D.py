import deeplabcut as dlc
import os
import datetime
import shutil
import sys
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

TF_FORCE_GPU_ALLOW_GROWTH = True
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


class Process3D:

    def __init__(self, file_3d, ground_file_3d, cal_dir, vid_dir):
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
        self.ground_net = ground_file_3d
        from tensorflow.python.client import device_lib
        device_lib.list_local_devices()
        self.repair_ground()

    def repair_ground(self):
        for viddate in self.videos:
            try:
                ground_vids = os.listdir(os.path.join(self.vid_dir, viddate, 'ground'))
            except FileNotFoundError:
                continue
            for ground_vid in ground_vids:
                try:
                    if 'camera-1' in ground_vid:
                        os.rename(os.path.join(self.vid_dir, viddate, 'ground', ground_vid),
                                  os.path.join(self.vid_dir, viddate, 'ground', 'camera-1_ground.avi'))
                    elif 'camera-2' in ground_vid:
                        os.rename(os.path.join(self.vid_dir, viddate, 'ground', ground_vid),
                                  os.path.join(self.vid_dir, viddate, 'ground', 'camera-2_ground.avi'))
                except FileExistsError:
                    pass

    def calibrate(self, cal_date_dir):
        # move cal images to 3D project
        try:
            shutil.rmtree(os.path.join(self.net, 'calibration_images'))
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree(os.path.join(self.ground_net, 'calibration_images'))
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree(os.path.join(self.net, 'camera_matrix'))
            os.mkdir(os.path.join(self.net, 'camera_matrix'))
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree(os.path.join(self.ground_net, 'camera_matrix'))
            os.mkdir(os.path.join(self.ground_net, 'camera_matrix'))
        except FileNotFoundError:
            pass
        shutil.copytree(os.path.join(self.cal_dir, cal_date_dir), os.path.join(self.net, 'calibration_images'))
        shutil.copytree(os.path.join(self.cal_dir, cal_date_dir), os.path.join(self.ground_net, 'calibration_images'))
        net_file = os.path.join(self.net, 'config.yaml')
        ground_net_file = os.path.join(self.ground_net, 'config.yaml')
        dlc.calibrate_cameras(net_file, cbrow=8, cbcol=8, calibrate=True, alpha=0.9)
        dlc.calibrate_cameras(ground_net_file, cbrow=8, cbcol=8, calibrate=True, alpha=0.9)
        print('************\n\ncalibration complete\n\n************')

    def triangulate(self, vid_date_dir):
        net_file = os.path.join(self.net, 'config.yaml')
        ground_net_file = os.path.join(self.ground_net, 'config.yaml')
        dlc.triangulate(net_file, os.path.join(self.vid_dir, vid_date_dir), filterpredictions=True, save_as_csv=True)
        dlc.triangulate(ground_net_file, os.path.join(self.vid_dir, vid_date_dir, 'ground'), filterpredictions=True, save_as_csv=True)
        print("************\n\ntriangulation complete\n\n************")
        try:
            dlc.create_labeled_video_3d(config=net_file,
                                        path=[os.path.join(self.vid_dir, vid_date_dir)])
            dlc.create_labeled_video_3d(config=ground_net_file,
                                        path=[os.path.join(self.vid_dir, vid_date_dir)])
        except Exception:
            "skipping labelled frame generation, data still available"
            return

        print("************\n\nFinished generating labelled clips************\n\n")

    def process(self):
        for vid_date in self.videos:
            if '07242020' in vid_date or '07252020' in vid_date:
                continue
            try:
                ground_vids = os.listdir(os.path.join(self.vid_dir, vid_date, 'ground'))
            except FileNotFoundError:
                print('skipping ' + vid_date)
                continue
            date = str(vid_date[12:20])
            vdate = datetime.date(month=int(date[0:2]),
                                  day=int(date[2:4]),
                                  year=int(date[4:8]))
            best_cal = None
            best_date = datetime.date(2000, 1, 1)
            for c in self.cal:
                cdate = str(c[5:])
                cdate = datetime.date(month=int(cdate[0:2]),
                                      day=int(cdate[2:4]),
                                      year=int(cdate[4:8]))
                if vdate > cdate > best_date:
                    best_cal = c
                    best_date = cdate
            if best_cal is not None:
                print("************\n\nProcessing " + vid_date + " using cal file from " + str(best_date) + '\n\n************')
                self.calibrate(best_cal)
                self.triangulate(vid_date)


if __name__ == "__main__":
    net_path = sys.argv[1]
    ground_net_path = sys.argv[2]
    cal_dir_path = sys.argv[3]
    vid_dir_path = sys.argv[4]
    processor = Process3D(net_path, ground_net_path, cal_dir_path, vid_dir_path)
    processor.process()
