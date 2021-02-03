import numpy as np
from scipy.stats import zscore
import cv2
from cv2 import VideoCapture
import os.path
import pickle as pkl
import math
import matplotlib.pyplot as plt
import sys
from sklearn.ensemble import GradientBoostingClassifier


class Recognizer:
    """
    Class to produce snip videos from two cameras where, as well as list of trial start end times.
    Built to always recognize the first frame of trial. Can pass any pretrained sklearn classifier
    type object.
    """
    def __init__(self, dir_path, video1_name, video2_name, classifier_object_path, frame_rate):
        # get video frame object
        # dictionary of crop dimensions for each video, toFlip, use clfObj, naming convention box_id + '_' + cam number
        ratio_dict = {
            'box2': {'533': ((50, 368, 250, 568), True, True),
                       '839': ((0, 318, 200, 518), True, True)},
            'box1': {'823': ((13, 375, 38, 440), False, False),
                       '889': ((71, 495, 124, 545), True, False)}
            }
        box_dict = {'mitg05': 'box2',
                         'mitg04': 'box1',
                         'mitg10': 'box1',
                         'mitg12': 'box1'}
        self.mouse_id = video1_name[0:5]
        if self.mouse_id != video2_name[0:5]:
            raise (ValueError, "videos must be from same mouse!")
        self.box_name = box_dict[video1_name[0:6]]
        if self.box_name != box_dict[video2_name[0:6]]:
            raise ValueError("both videos must be recorded on same box")
        self.cam1_id = video1_name[7:10]
        self.cam2_id = video2_name[7:10]
        self.vid1_preset = ratio_dict[self.box_name][self.cam1_id]
        self.vid2_preset = ratio_dict[self.box_name][self.cam2_id]

        self.video_id = video1_name[12:-4]

        with open(classifier_object_path, 'rb') as f:
            self.clf = pkl.load(f)

        self.Xarrs = [None, None]
        self.Xarrs[0] = self._extract_frames(os.path.join(dir_path, video1_name), flip=self.vid1_preset[1], crop_to=self.vid1_preset[0])
        self.Xarrs[1] = self._extract_frames(os.path.join(dir_path, video2_name), flip=self.vid2_preset[1], crop_to=self.vid2_preset[0])

        self.trial_times = [np.empty(0), np.empty(0)]  # init as np none to prevent compiler warnings
        self.frame_rate = frame_rate

    def _extract_frames(self, video_path, flip: bool, crop_to: tuple):
        vidcap = VideoCapture(video_path)
        if not vidcap.isOpened():
            raise ValueError('Video could not be opened.')
        # n = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        n = 10000
        h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        X = np.zeros((n, h, w), dtype=np.uint8)
        success, image = vidcap.read()
        count = 0
        while success and count < n:
            image = image[:, :, 0]  # discard redundant channels
            if flip:
                image = cv2.flip(image, 0)
            X[count, :, :] = image
            count += 1
            success, image = vidcap.read()
        return X

    def _preprocess(self, vid_arr, crop_to: tuple):
        fin_w = round((crop_to[3] - crop_to[1]) / 4)
        fin_h = round(
            (crop_to[2] - crop_to[0]) / 4)  # set width and height of video frames, must match training data
        if fin_w < 1 or fin_h < 1:
            raise IndexError("dims must be positive.")
        nx = np.zeros((vid_arr.shape[0], fin_h, fin_w), dtype=np.uint8)
        for i, frame in enumerate(vid_arr):
            image = frame[crop_to[0]:crop_to[2], crop_to[1]:crop_to[3]]  # crop

            image = cv2.resize(image, (0, 0),
                               fx=.25,
                               fy=.25,
                               interpolation=cv2.INTER_NEAREST)
            nx[i] = image
        return nx

    def _default_clf(self, X, threshold=2.):
        means = np.mean(np.mean(X, axis=1), axis=1)
        z = zscore(means)
        z[z <= threshold] = 0
        z[z > threshold] = 1
        return z

    def _get_predict_individual(self):
        """
        For to be considered trial, prev second must be mostly 0s and next must be 5
        frames of 1s
        :return:
        """
        crop_to = [self.vid1_preset[0], self.vid2_preset[0]]
        for i in range(2):
            nx = self._preprocess(self.Xarrs[i], crop_to[i])
            if self.vid1_preset[2]:
                yhat = self.clf.predict(nx.reshape((nx.shape[0], -1)))
            else:
                if i == 0:
                    yhat = self._default_clf(nx, threshold=2)
                else:
                    yhat = self._default_clf(nx, threshold=2.5)
            # Smooth, extract trial times
            trials = []
            trial_counter = 0
            j = self.frame_rate
            while j < yhat.shape[0] - (self.frame_rate * 2):
                prev = yhat[j - self.frame_rate: j]
                next = yhat[j: j + 4]
                if np.mean(prev) <= .1 and np.mean(next) >= 1:
                    # trial found
                    trials.append([j - self.frame_rate, j + (self.frame_rate * 2)])
                    trial_counter += 1
                    j += (self.frame_rate * 2)
                else:
                    j += 1
            trials = np.array(trials, dtype=np.uint64)
            self.trial_times[i] = trials


    def predict(self):
        """
        Must solve the problem of aligning videos.
        :return:
        """
        self._get_predict_individual()
        # assume first trial is discovered
        final_trial_times1 = [self.trial_times[0][0]]
        final_trial_times2 = [self.trial_times[1][0]]

        full_lens = [None, None]
        full_lens[0] = self.trial_times[0].shape[0]
        full_lens[1] = self.trial_times[1].shape[0]

        if full_lens[0] != full_lens[1]:
            print("WARNING: Different number of trials detected across cameras. Attempting to resolve...", sys.stderr)

        inter_lens = [None, None]
        inter_lens[0] = np.diff(self.trial_times[0][:, 0])
        inter_lens[1] = np.diff(self.trial_times[1][:, 0])

        tol = self.frame_rate
        # align the two cameras
        has_next = True
        cam1_ind = 0
        cam2_ind = 0
        c1_dist = 0
        c2_dist = 0
        while has_next:
            c1_dist += inter_lens[0][cam1_ind]
            c2_dist += inter_lens[1][cam2_ind]
            if math.isclose(c1_dist, c2_dist, abs_tol=tol):
                c1_dist = 0
                c2_dist = 0
                final_trial_times1.append(self.trial_times[0][cam1_ind + 1])
                final_trial_times2.append(self.trial_times[1][cam2_ind + 1])
                cam1_ind += 1
                cam2_ind += 1
            elif c1_dist > c2_dist:
                # c1 missed this trial, skip it for c2 as well
                cam2_ind += 1
                c1_dist = 0
            elif c2_dist > c1_dist:
                # c2 missed this trial, skip it for c1 as well
                cam1_ind += 1
                c2_dist = 0
            if cam1_ind == inter_lens[0].shape[0] - 1 or cam2_ind == inter_lens[1].shape[0] - 1:
                has_next = False
        return final_trial_times1, final_trial_times2

    def write_clips(self, trial_times, cam_num, base_path):
        snip_dir = os.path.join(base_path, self.box_name + '_' + self.mouse_id + '_' + self.video_id)
        try:
            os.mkdir(snip_dir)
        except FileExistsError:
            pass
        for i, times in enumerate(trial_times):
            out = cv2.VideoWriter(os.path.join(snip_dir, 'camera-' + str(cam_num) + '_clip-' + str(i) + '.avi'),
                                  cv2.VideoWriter_fourcc('M','P','E','G'), 30, (self.Xarrs[cam_num].shape[1], self.Xarrs[cam_num].shape[2]))
            if times[1] > self.Xarrs[cam_num].shape[0]:
                times[1] = self.Xarrs[cam_num].shape[0] - 1
            vid_frame = self.Xarrs[cam_num][times[0]:times[1]]
            for frame in vid_frame:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out.write(frame)
            out.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    # quick test code
    # recog = Recognizer(dir_path='/home/spencerloggia/Documents/',
    #                    video1_name='mitg12-823--07042020111739.avi',
    #                    video2_name='mitg12-889--07042020111738.avi',
    #                    classifier_object_path='./SVClassifier.pkl',
    #                    frame_rate=30)
    #
    # f = open('./recog_dump.pkl', 'wb')
    # pkl.dump(recog, f)

    f = open('./recog_dump.pkl', 'rb')
    recog = pkl.load(f)

    recog.frame_rate = 30
    final_trial_times1, final_trial_times2 = recog.predict()
    recog.write_clips(final_trial_times1, 0, './')
    recog.write_clips(final_trial_times2, 1, './')
    print('done.')


