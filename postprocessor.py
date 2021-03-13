import pandas as pd
import numpy as np
import cv2
import os

class Postprocessor:

    def __init__(self, traj_dir:str, num_bodyparts):
        ground_data = os.listdir(os.path.join(traj_dir, 'ground'))
        df = None
        for file in ground_data:
            if '_3D' in file:
                # found the ground 3D file.
                df = pd.read_csv(file)
        if df is None:
            raise RuntimeError
