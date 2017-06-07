from __future__ import print_function

import glob
import random
import os
import copy
from itertools import product

SEED = 7687655
import numpy as np
np.random.seed(SEED)
random.seed(SEED)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid': False})

from sklearn.feature_extraction.image import extract_patches_2d

from skimage import io
from skimage.transform import resize, downscale_local_mean
from scipy.misc import imsave

from keras.utils import np_utils

import augment

AUGMENTATION_PARAMS = {
    'zoom_range': (0.75, 1.25),
    'rotation_range': (-10, 10),
    'shear_range': (-10, 10),
    'translation_range': (-12, 12),
    'do_flip': False,
    'allow_stretch': False,
}

NO_AUGMENTATION_PARAMS = {
    'zoom_range': (1.0, 1.0),
    'rotation_range': (0, 0),
    'shear_range': (0, 0),
    'translation_range': (0, 0),
    'do_flip': False,
    'allow_stretch': False,
}


def load_dir(dir_name, ext='.tif', max_nb=None,
             script_classes=None):
    X, scripts, dates = [], [], []
    fns = glob.glob(dir_name+'/*' + ext)
    random.shuffle(fns)

    for fn in fns:
        if max_nb and len(X) >= max_nb:
            break
        scr, dat, _ = os.path.basename(fn).split('-')[:3]
        if script_classes and scr not in script_classes:
            continue
        else:
            img = np.array(io.imread(fn), dtype='float32')
            scaled = img / np.float32(255.0)
            scaled = downscale_local_mean(scaled, factors=(2, 2))
            X.append(scaled)
            dates.append(dat)
            scripts.append(scr)

    return X, scripts, dates


def augment_images(images, scripts, dates, nb_rows, nb_cols,
                   nb_patches, distort=True):
    X, script_Y, date_Y = [], [], []
    for idx, (image, script, date) in enumerate(zip(images, scripts, dates)):
        patches = extract_patches_2d(image=image,
                                     patch_size=(nb_rows * 2, nb_cols * 2),
                                     max_patches=nb_patches)
        for patch in patches:
            if distort:
                patch = augment.perturb(patch, AUGMENTATION_PARAMS,
                                        target_shape=(nb_rows, nb_cols))
            else:
                patch = augment.perturb(patch, NO_AUGMENTATION_PARAMS,
                                        target_shape=(nb_rows, nb_cols))
            patch = patch.reshape((1, patch.shape[0], patch.shape[1]))
            X.append(patch)
            script_Y.append(script)
            date_Y.append(date)

    X = np.array(X, dtype='float32')
    script_Y = np.array(script_Y, dtype='int32')
    date_Y = np.array(date_Y, dtype='int32')

    return X, script_Y, date_Y
