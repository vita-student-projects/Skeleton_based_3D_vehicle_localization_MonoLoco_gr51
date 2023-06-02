
import math
from copy import deepcopy

import numpy as np

from ..utils import correct_angle, to_cartesian, to_spherical

BASELINE = 0.54
BF = BASELINE * 721

COCO_KEYPOINTS =  [
    'front_up_right',       # 1
    'front_up_left',        # 2
    'front_light_right',    # 3
    'front_light_left',     # 4
    'front_low_right',      # 5
    'front_low_left',       # 6
    'central_up_left',      # 7
    'front_wheel_left',     # 8
    'rear_wheel_left',      # 9
    'rear_corner_left',     # 10
    'rear_up_left',         # 11
    'rear_up_right',        # 12
    'rear_light_left',      # 13
    'rear_light_right',     # 14
    'rear_low_left',        # 15
    'rear_low_right',       # 16
    'central_up_right',     # 17
    'rear_corner_right',    # 18
    'rear_wheel_right',     # 19
    'front_wheel_right',    # 20
    'rear_plate_left',      # 21
    'rear_plate_right',     # 22
    'mirror_edge_left',     # 23
    'mirror_edge_right',    # 24
]

HFLIP = {
    'front_up_right': 'front_up_left',
    'front_light_right': 'front_light_left',
    'front_low_right': 'front_low_left',
    'central_up_left': 'central_up_right',
    'front_wheel_left': 'front_wheel_right',
    'rear_wheel_left': 'rear_wheel_right',
    'rear_corner_left': 'rear_corner_right',
    'rear_up_left': 'rear_up_right',
    'rear_light_left': 'rear_light_right',
    'rear_low_left': 'rear_low_right',
    'front_up_left': 'front_up_right',
    'front_light_left': 'front_light_right',
    'front_low_left': 'front_low_right',
    'central_up_right': 'central_up_left',
    'front_wheel_right': 'front_wheel_left',
    'rear_wheel_right': 'rear_wheel_left',
    'rear_corner_right': 'rear_corner_left',
    'rear_up_right': 'rear_up_left',
    'rear_light_right': 'rear_light_left',
    'rear_low_right': 'rear_low_left',
    'rear_plate_left': 'rear_plate_right',
    'rear_plate_right': 'rear_plate_left',
    'mirror_edge_left': 'mirror_edge_right',
    'mirror_edge_right': 'mirror_edge_left'
}

def transform_keypoints(keypoints, mode):
    """Egocentric horizontal flip"""
    assert mode == 'flip', "mode not recognized"
    kps = np.array(keypoints)
    dic_kps = {key: kps[:, :, idx] for idx, key in enumerate(COCO_KEYPOINTS)}
    kps_hflip = np.array([dic_kps[value] for key, value in HFLIP.items()])
    kps_hflip = np.transpose(kps_hflip, (1, 2, 0))
    return kps_hflip.tolist()


def flip_inputs(keypoints, im_w, mode=None):
    """Horizontal flip the keypoints or the boxes in the image"""
    if mode == 'box':
        boxes = deepcopy(keypoints)
        for box in boxes:
            temp = box[2]
            box[2] = im_w - box[0]
            box[0] = im_w - temp
        return boxes

    keypoints = np.array(keypoints)
    keypoints[:, 0, :] = im_w - keypoints[:, 0, :]  # Shifted
    kps_flip = transform_keypoints(keypoints, mode='flip')
    return kps_flip


def flip_labels(boxes_gt, labels, im_w):
    """Correct x, d positions and angles after horizontal flipping"""
    boxes_flip = deepcopy(boxes_gt)
    labels_flip = deepcopy(labels)

    for idx, label_flip in enumerate(labels_flip):

        # Flip the box and account for disparity
        disp = BF / label_flip[2]
        temp = boxes_flip[idx][2]
        boxes_flip[idx][2] = im_w - boxes_flip[idx][0] + disp
        boxes_flip[idx][0] = im_w - temp + disp

        # Flip X and D
        rtp = label_flip[3:4] + label_flip[0:2]  # Originally t,p,z,r
        xyz = to_cartesian(rtp)
        xyz[0] = -xyz[0] + BASELINE  # x
        rtp_r = to_spherical(xyz)
        label_flip[3], label_flip[0], label_flip[1] = rtp_r[0], rtp_r[1], rtp_r[2]

        # FLip and correct the angle
        yaw = label_flip[9]
        yaw_n = math.copysign(1, yaw) * (np.pi - abs(yaw))  # Horizontal flipping change of angle

        sin, cos, _ = correct_angle(yaw_n, xyz)
        label_flip[7], label_flip[8], label_flip[9] = sin, cos, yaw_n

    return boxes_flip, labels_flip


def height_augmentation(kps, kps_r, label_s, seed=0):
    """
    label_s: theta, psi, z, rho, wlh, sin, cos, s_match
    """
    n_labels = 3 if label_s[-1] > 0.9 else 1
    height_min = 1.2
    height_max = 2
    av_height = 1.5
    kps_aug = [[kps.clone(), kps_r.clone()] for _ in range(n_labels+1)]
    labels_aug = [label_s.copy() for _ in range(n_labels+1)]  # Maintain the original
    np.random.seed(seed)
    heights = np.random.uniform(height_min, height_max, n_labels)  # 3 samples
    zzs = heights * label_s[2] / av_height
    disp = BF / label_s[2]

    rtp = label_s[3:4] + label_s[0:2]  # Originally t,p,z,r
    xyz = to_cartesian(rtp)

    for i in range(n_labels):

        if zzs[i] < 2:
            continue
        # Update keypoints
        disp_new = BF / zzs[i]
        delta_disp = disp - disp_new
        kps_aug[i][1][0, 0, :] = kps_aug[i][1][0, 0, :] + delta_disp

        # Update labels
        labels_aug[i][2] = zzs[i]
        xyz[2] = zzs[i]
        rho = np.linalg.norm(xyz)
        labels_aug[i][3] = rho

    return kps_aug, labels_aug
