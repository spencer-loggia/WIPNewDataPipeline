import matplotlib.pyplot as plt
import cv2
from cv2 import VideoCapture
import os
import sys
import trialRecognition

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

RATIO_DICT = {
    '2_2': (50, 400, 250, 200),
    '2_1': (0, 450, 200, 250),
    '1_1': (0, 450, 225, 225),
    '1_2': (0, 450, 225, 225)
}

def extract_frames_labels():
    # video preprocessing
    gt_file = open('data/gt.csv', 'w')
    gt_file.write('im_path,label\n')
    path = '/home/spencerloggia/Documents/stimuli_labelling_data'
    files = os.listdir(path)
    fnum = 0
    for file in files:
        fnum += 1
        count = 0
        full_path = os.path.join(path, file)
        vidcap = VideoCapture(full_path)
        success, image = vidcap.read()
        isneg = "neg" in file
        # extract 50 frames from each video
        while success:
            id = file[0:3]
            if id in RATIO_DICT:
                rat_tup = RATIO_DICT[id]
                image = image[rat_tup[0]:rat_tup[0] + 318, rat_tup[2]:rat_tup[2] + 318, 0]  # discard redundant channels and crop
            image = cv2.resize(image, (0, 0),
                               fx=.25,
                               fy=.25,
                               interpolation=cv2.INTER_NEAREST)  # downsample
            if isneg and count % 150 == 0:
                fname = "%d_frame_%d.jpg" % (fnum, count)
                cv2.imwrite('data/' + fname, image)
                gt_file.write(fname + ',0' + '\n')
            elif not isneg and count > 15 and count % 10 == 0:
                fname = "%d_frame_%d.jpg" % (fnum, count)
                gt_file.write(fname + ',1' + '\n')
                cv2.imwrite('data/' + fname, image)

            count += 1
            success, image = vidcap.read()
    gt_file.close()


if __name__ == '__main__':
    if sys.argv[1] == 'gen_data':
        extract_frames_labels()
    elif sys.argv[1] == 'train':
        pass
        #TODO: add this code to this version
    elif sys.argv[1] == 'predict':
        dir = sys.argv[2]
        fname1 = sys.argv[3]
        fname2 = sys.argv[4]
        clf_path = sys.argv[5]
        fr = int(sys.argv[6])
        sheet = sys.argv[7]
        recog = trialRecognition.Recognizer(dir,
                           fname1,
                           fname2,
                           clf_path,
                           fr,
                           sheet)
        disc1, disc2 = recog.predict()

