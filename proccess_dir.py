from trialRecognition import Recognizer
import sys
import os

#define camera 1 and 2::
CAM_DEF = {
    '823': (1, 30),
    '889': (2, 30),
    '533': (2, 50),
    '839': (1, 50)
}


def _connect_recognizer(vid_dir, out_dir, vid_a, vid_b):
    info_a = CAM_DEF[vid_a[7:10]]
    info_b = CAM_DEF[vid_b[7:10]]

    if info_a[0] == 1 and info_b[0] == 2:
        recog = Recognizer(vid_dir, vid_a, vid_b, info_a[1])
    elif info_a[0] == 2 and info_b[0] == 1:
        recog = Recognizer(vid_dir, vid_b, vid_a, info_a[1])
    else:
        print('Error')
        return
    try:
        final_trial_times1, final_trial_times2 = recog.predict()
    except RuntimeError:
        print("failed to match " + str(vid_a) + " and " + str(vid_b), sys.stderr)
        return
    recog.write_clips(final_trial_times1, 0, out_dir)
    recog.write_clips(final_trial_times2, 1, out_dir)


def process(vid_dir_path, output_dir):
    vids = set(os.listdir(vid_dir_path))
    while len(vids) > 1:
        cur = vids.pop()
        date = cur[12:20]
        match = None
        for poss_match in vids:
            if poss_match[12:20] == date:
                match = poss_match
                break
        if match is not None:
            vids.remove(match)
            _connect_recognizer(vid_dir_path, output_dir, cur, match,)


if __name__ == "__main__":
    vid_dir_path = sys.argv[1]  # directory containing all input videos
    output_dir = sys.argv[2]  # location where folders for each days snips will be stored
    process(vid_dir_path, output_dir)
