import os
from cv2 import VideoCapture
import cv2
import numpy as np
from scipy.stats import zscore


class CalPreprocessor:
    """
    class for extracting matching frame pairs from calibration videos.
    Much simpler than the stimuli recognition script. Searches for a
    region of video that is consistently different (as determined by Z-score
    from the rest of the video and treats that as the offset. Then can extract
    matching frame pairs based on offset.
    """
    def __init__(self, cal_file_1: str, cal_file_2: str, output_dir: str, num_frame_pairs=60):
        dir1, cal1 = os.path.split(cal_file_1)
        dir2, cal2 = os.path.split(cal_file_2)

        self.num_frame_pairs = num_frame_pairs
        self.output_dir = output_dir

        # first 2 are box 2, second 2 are box 1
        self.flip_dict = {
            '533': True,
            '839': True,
            '823': False,
            '889': True
        }

        if cal1[9:16] != cal2[9:16]:
            raise Exception("Cal files must be from same day")

        self.video1 = cal_file_1
        self.video2 = cal_file_2

    def extract_frames(self, video_path, flip: bool):
        vidcap = VideoCapture(video_path)
        n = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        X = np.zeros((n, h, w), dtype=np.uint8)
        success, image = vidcap.read()
        count = 0
        while success and count < n:
            if flip:
                image = cv2.flip(image, 0)
            X[count, :, :] = image[:, :, 1]
            count += 1
            success, image = vidcap.read()
        return X

    def _find_indicator(self, X):
        means = np.mean(np.mean(X, axis=1), axis=1)
        z = zscore(means)
        prev = [0] * 10
        i = 0
        while i < means.shape[0]:
            prev.pop(0)
            prev.append(np.abs(z[i]))
            if np.min(prev) >= 3.5:
                # flash found
                flash_ind = i - 9
                return flash_ind
            i += 1
        raise Exception("The flash could not be found.")

    def save_matched_sample(self):
        cal1_cam = os.path.basename(self.video1)[4:7]
        cal2_cam =  os.path.basename(self.video2)[4:7]

        X1 = self.extract_frames(self.video1, self.flip_dict[cal1_cam])
        X2 = self.extract_frames(self.video2, self.flip_dict[cal2_cam])

        t1 = self._find_indicator(X1)
        t2 = self._find_indicator(X2)

        offset = t2 - t1  # how much is 2 ahead of 1

        sample = np.random.random_integers(np.abs(offset),
                                           X1.shape[0] - np.abs(offset),
                                           self.num_frame_pairs)

        for ind in sample:
            fname1 = "camera-1_image_%d.jpg" % ind
            cv2.imwrite(os.path.join(self.output_dir, fname1), X1[ind])
            fname2 = "camera-2_image_%d.jpg" % ind
            cv2.imwrite(os.path.join(self.output_dir, fname2), X2[ind+offset])





