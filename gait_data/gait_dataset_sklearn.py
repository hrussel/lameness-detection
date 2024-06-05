import os
import numpy as np
import gait_data.gait_data
from sklearn.utils import Bunch
import core.config
import matplotlib.pyplot as plt

# Feature names consts
BPM = "BPM"
HBA = "HBA"
HBB = "HBB"
HBB_H = "HBB_H"
HBB_F = "HBB_F"
TRK = "TRK"
TRK_L = "TRK_L"
TRK_R = "TRK_R"
STL = "STL"
STL_R = "STL_R"
STD = "STD"
SWD = "SWD"


def load_gait_dataset(config):

    gait_scores_np = np.genfromtxt(config.gait_scores_csv,
                                   dtype=str,
                                   delimiter=',',
                                   skip_header=True,
                                   usecols=[0, 1, 2]  # Video, ID, Menno
                                   )
    data = []
    labels = []
    ids = []
    feature_names = None
    video_names = []
    merging = config.merging
    thr = int(merging.split('-')[-1][0])  # first digit in the last group of merging.
    binary = False
    if len(merging.split('-')) == 2:
        binary = True

    for video, id, score in gait_scores_np:
        score = int(score)

        if score > thr:  # merge the score
            score = thr

        # binarize score
        score -= 1

        kp_csv = os.path.join(config.keypoints_path, "%s.csv" % video)
        gd = gait_data.gait_data.GaitData(video, kp_csv, config.joints, cow_id=id)

        # If we don't have at least 2 steps for each leg, discard video
        min_steps, max_steps = gd.min_max_steps()
        if min_steps < 2:
            print(f"Not enough steps in video {video}. Skipping.")
            continue

        features, feature_names = add_features(config, gd)
        for ff, feat in enumerate(features):
            if feat is None or np.isnan(feat):
                print(video, feature_names[ff])

        data.append(features)

        labels.append(score)
        ids.append(int(id))
        video_names.append(video)

    data = np.array(data)
    labels = np.array(labels)
    ids = np.array(ids)
    video_names = np.array(video_names)
    feature_names = np.array(feature_names)

    return Bunch(data=data, labels=labels, feature_names=feature_names, ids=ids, video_names=video_names)


def add_features(config, gait_data):
    features = []
    feature_names = []

    if BPM in config.features:
        bpm = gait_data.compute_back_curvature(method='BPM')
        features.append(bpm)
        feature_names.append(BPM)

    if HBA in config.features:
        hba = gait_data.head_amplitude()
        features.append(hba)
        feature_names.append(HBA)

    if HBB in config.features:
        hbb = gait_data.head_bobbing()

        features.append(hbb['H'])
        feature_names.append(HBB+'_H')
        features.append(hbb['F'])
        feature_names.append(HBB+'_F')

    if TRK in config.features:
        trk = gait_data.tracking_distance()

        features.append(trk['L'])
        feature_names.append(TRK+'_L')
        features.append(trk['R'])
        feature_names.append(TRK+'_R')

    if STL in config.features:
        stl = gait_data.stride_length(mean=True)

        for k in stl:
            features.append(stl[k])
            feature_names.append(f'{STL}_{k}')

    if STL_R in config.features:
        stl = gait_data.stride_length2()

        for k in stl:
            features.append(stl[k])
            feature_names.append(f'{STL}_{k}')

    if STD in config.features:

        std = gait_data.stance_duration_difference(fps=1)

        for k in std:
            features.append(std[k])
            feature_names.append(f'{STD}_{k}')

    if SWD in config.features:

        swd = gait_data.swing_duration_difference(fps=1)

        for k in swd:
            features.append(swd[k])
            feature_names.append(f'{SWD}_{k}')

    return features, feature_names


class MyArgsConfig:
    def __init__(self, config_path):
        self.config = config_path


if __name__ == '__main__':

    args = MyArgsConfig("../cfg/config-desktop.yml")
    config = core.config.MyConfig(args)
    figs = load_gait_dataset(config)
