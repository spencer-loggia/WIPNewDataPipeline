from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import cv2
from cv2 import VideoCapture
import os.path
import pickle as pkl
import math
import pandas as pd

class Recognizer:
    """
    Class to produce snip videos from two cameras where, as well as list of trial start end times.
    Built to always recognize the first frame of trial. Can pass any pretrained sklearn classifier
    type object.
    """
    def __init__(self, dir_path, video1_name, video2_name, classifier_object_path, frame_rate, sheet_path):
        # get video frame object
        # dictionary of crop dimensions for each video, naming convention box_id + '_' + cam number
        ratio_dict = {
            'box2': {'533': ((50, 400, 250, 200), True),
                       '839': ((0, 450, 200, 250), True)},
            'box1': {'823': ((0, 450, 225, 225), False),
                       '889': ((0, 450, 225, 225), True)}
            }
        box_dict = {'mitg05': 'box1',
                         'mitg04': 'box1',
                         'mitg10': 'box1'}
        if video1_name[0:5] != video2_name[0:5]:
            raise (ValueError, "videos must be from same mouse!")
        vid1_preset = ratio_dict[box_dict[video1_name[0:5]]][video1_name[7:9]]
        vid2_preset = ratio_dict[box_dict[video2_name[0:5]]][video2_name[7:9]]

        with open(classifier_object_path, 'rb') as f:
            self.clf = pkl.load(f)

        self.Xarrs = [np.empty(0), np.empty(0)]
        self.Xarrs[0] = self._extract_frames(os.path.join(dir_path, video1_name), flip=vid1_preset[1], crop_to=vid1_preset[0])
        self.Xarrs[1] = self._extract_frames(os.path.join(dir_path, video2_name), flip=vid2_preset[1], crop_to=vid2_preset[0])

        self.trial_times = [np.empty(0), np.empty(0)]  # init as np none to prevent compiler warnings
        self.sheet_data = self._load_sheet_data(sheet_path)
        self.frame_rate = frame_rate


    def _extract_frames(self, video_path, flip: bool, crop_to: tuple):
        vidcap = VideoCapture(video_path)
        n = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fin_w = fin_h = 80  # set width and height of video frames, must match training data
        X = np.zeros((n, fin_h, fin_w), dtype=np.uint8)
        success, image = vidcap.read()
        count = 0
        while success and count < n:
            if flip:
                image = cv2.flip(image, 0)
            image = image[crop_to[0]:crop_to[0] + 318, crop_to[2]:crop_to[2] + 318, 0]  # discard redundant channels and crop
            image = cv2.resize(image, (0, 0),
                               fx=.25,
                               fy=.25,
                               interpolation=cv2.INTER_NEAREST)  # downsample
            X[count, :, :] = image
            count += 1
            success, image = vidcap.read()
        return X

    def _get_predict_individual(self):
        for i in range(2):
            yhat = self.clf.predict(self.Xarrs[i])
            # Smooth, extract trial times
            prev = [0] * 20
            trials = []
            trial_counter = 0
            j = 20
            innum = False
            while j < yhat.shape[0]:
                prev.pop(0)
                prev.append(yhat[j])
                if not innum and np.mean(prev) >= .85 and 0 not in prev[j - 20:j - 17]:
                    # trial found
                    innum = True
                    trials.append([j - 20, -1])
                    trial_counter += 1
                if innum and np.mean(prev) < .15 and 1 not in yhat[j - 20:i - 17]:
                    innum = False
                    trials[-1][1] = j - 20
                j += 1
            trials = np.array(trials, dtype=np.uint64)
            self.trial_times[i] = trials

    def _load_sheet_data(self, path) -> np.ndarray[int]:
        f = open(path, 'r')
        count = 0

        # determine beginning of relevent data.
        for line in f:
            line = line.split()
            if line[0] == 'Trial':
                break
            count += 1
        data = pd.read_csv(path, skiprows=count, skipfooter=1)
        response_times = data['entry.1', 'exit.1'].to_numpy()
        response_times *= self.frame_rate # convert to frames.
        return response_times

    def _align_to_klimbic(self, vid_lens, sheet_lens):
        """
        align a cameras times to spreadsheet.
        :param vid_lens:
        :return: array of length spreadsheet data, with -1 where no match found and the index of the
                  video trial in the other positions.
        """
        alignment = np.array(sheet_lens.shape[0])
        vid_ind = 0
        sheet_ind = 0
        while vid_ind < vid_lens.shape[0] and sheet_ind < sheet_lens.shape[0]:
            if math.isclose(vid_lens[vid_ind], sheet_lens[sheet_ind], abs_tol=3):
                alignment[sheet_ind] = vid_ind
                vid_ind += 1
            else:
                alignment[sheet_ind] = -1
            sheet_ind += 1
        return alignment

    def predict(self):
        """
        Must solve the problem of aligning three sequences. Luckily, optimal overlap alignment is
        not nessesary, we are ok with throwing out poor matches, and are primarily concerned with
        preserving clips we have high confidence in (e.g. near length match between all three data-sources,
        and close in local time.)
        :return:
        """
        lens = [None, None]
        lens[0] = np.diff(self.trial_times[0], axis=1)
        lens[1] = np.diff(self.trial_times[0], axis=1)
        sheet_lens = np.diff(self.sheet_data, axis=1)

        vid1_sheet_align = self._align_to_klimbic(lens[0], sheet_lens)
        vid2_sheet_align = self._align_to_klimbic(lens[1], sheet_lens)

        discoverd_vid1 = []
        discoverd_vid2 = []

        max_allowed_discovery_diff = np.abs(lens[0].shape[0] - lens[1].shape[0])
        # align the two cameras
        for i in range(sheet_lens.shape[0]):
            if vid1_sheet_align[i] != -1 and vid2_sheet_align[i] != -1:
                diff = np.abs(vid1_sheet_align[i] - vid2_sheet_align[i])
                max_allowed_discovery_diff -= diff
                if max_allowed_discovery_diff < 0:
                    break
                discoverd_vid1.append((self.trial_times[0][vid1_sheet_align[i]]))
                discoverd_vid2.append((self.trial_times[1][vid1_sheet_align[i]]))
        return discoverd_vid1, discoverd_vid2



