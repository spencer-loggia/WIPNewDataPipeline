import os
from cv2 import VideoCapture
import cv2
import numpy as np
from scipy.stats import zscore
import sys


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
            if np.min(prev) >= 2.5:
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


def batch_process(source_dir: str, target_dir: str, num_frames: int):
    # reads all cal videos in source dir and extracts frame pairs to target dir
    CAM_DEF = {
        '823': (1, 1, '889'),
        '889': (1, 2, '823'),
        '533': (2, 2, '839'),
        '839': (2, 1, '533')
    }  # (cam_id: (box#, cam#, partner_id)
    videos = set(os.listdir(source_dir))
    try:
        os.mkdir(target_dir)
    except FileExistsError:
        pass
    while len(videos) > 1:
        vid = videos.pop()
        cam_id = vid[4:7]
        date = vid[9:17]
        vid_match = None
        # find matching video
        partner_id = CAM_DEF[cam_id][2]
        for nvid in videos:
            ncam_id = nvid[4:7]
            ndate = nvid[9:17]
            if ncam_id == partner_id and ndate == date:
                vid_match = nvid
                break
        if vid_match is None:
            continue
        videos.remove(vid_match)
        box_num = CAM_DEF[cam_id][0]
        cam_num = CAM_DEF[cam_id][1]
        final_dir = os.path.join(target_dir, 'box' + str(box_num) + '_' + date)
        os.mkdir(final_dir)
        if cam_num == 1:
            # first vid found is camera 1
            vid1 = os.path.join(source_dir, vid)
            vid2 = os.path.join(source_dir, vid_match)
        else:
            # second found is camera 1
            vid1 = os.path.join(source_dir, vid_match)
            vid2 = os.path.join(source_dir, vid)
        print("proccessing " + vid + ' and ' + vid_match + ' ...')
        try:
            preprocessor = CalPreprocessor(vid1, vid2, final_dir, num_frames)
            preprocessor.save_matched_sample()
        except Exception:
            print("could not find flash, skipping...")
            os.rmdir(final_dir)

if __name__ == "__main__":
    source_dir = sys.argv[1]
    target_dir = sys.argv[2]
    num_frames = int(sys.argv[3])
    batch_process(source_dir, target_dir, num_frames)
