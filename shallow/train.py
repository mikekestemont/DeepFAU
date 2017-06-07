from __future__ import print_function

import random
import shutil
import os
import pickle

import sys
sys.setrecursionlimit(10000)

SEED = 1066987
import numpy as np
np.random.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from scipy.misc import imsave

import keras
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

import utils
from model import ShallowNet

BATCH_SIZE = 30
NB_EPOCHS = 300
NB_ROWS, NB_COLS = 150, 150
NB_TRAIN_PATCHES = 10
NB_TEST_PATCHES = 50
MODEL_NAME = 'SHALLOW+DEEP'

from keras.callbacks import ModelCheckpoint


def train():
    #l = {'4', '8'}
    l = None
    train_images, train_scripts, train_dates = \
        utils.load_dir('../data/splits/train',
                       max_nb=None,
                       script_classes=l)
    dev_images, dev_scripts, dev_dates = \
        utils.load_dir('../data/splits/dev',
                       max_nb=None,
                       script_classes=l)

    try:
        os.mkdir('models')
    except:
        pass

    try:
        shutil.rmtree('models/' + MODEL_NAME)
    except:
        pass

    os.mkdir('models/' + MODEL_NAME)

    script_encoder = LabelEncoder().fit(train_scripts)
    date_encoder = LabelEncoder().fit(train_dates)

    orig_train_script_int = script_encoder.transform(train_scripts)
    orig_train_date_int = date_encoder.transform(train_dates)
    orig_dev_script_int = script_encoder.transform(dev_scripts)
    orig_dev_date_int = date_encoder.transform(dev_dates)

    nb_scripts = len(script_encoder.classes_)
    nb_dates = len(date_encoder.classes_)

    print('-> Working on', nb_scripts,
          'script classes:', script_encoder.classes_)
    print('-> Working on', nb_dates,
          'date classes:', date_encoder.classes_)

    pickle.dump(script_encoder, open('models/' +
                MODEL_NAME + '/script_encoder.p', 'wb'))
    pickle.dump(date_encoder, open('models/' +
                MODEL_NAME + '/date_encoder.p', 'wb'))

    model = ShallowNet(nb_scripts=nb_scripts,
                       nb_dates=nb_dates,
                       nb_rows=NB_ROWS,
                       nb_cols=NB_COLS)

    model.summary()

    with open('models/' + MODEL_NAME + '/architecture.json', 'w') as F:
        F.write(model.to_json())

    best_dev_acc = 0.0

    print('-> cropping dev...')
    dev_crops = []
    for image in dev_images:
        Xs, _, _ = utils.augment_images(images=[image],
                                        scripts=[1],
                                        dates=[1],
                                        nb_rows=NB_ROWS,
                                        nb_cols=NB_COLS,
                                        nb_patches=NB_TEST_PATCHES,
                                        distort=False)
        dev_crops.append(Xs)

    example = dev_crops[0]
    for idx, crop in enumerate(example):
        crop = crop.reshape((crop.shape[1], crop.shape[2]))
        imsave(str(idx)+'_crop.png', crop)

    for e in range(NB_EPOCHS):
        print('-> in epoch', e)
        print('-> cropping train...')
        train_X, train_script_int, train_date_int = \
            utils.augment_images(images=train_images,
                                 scripts=orig_train_script_int,
                                 dates=orig_train_date_int,
                                 nb_rows=NB_ROWS,
                                 nb_cols=NB_COLS,
                                 nb_patches=NB_TRAIN_PATCHES,
                                 distort=True)
        train_scr_Y = np_utils.to_categorical(train_script_int,
                                              num_classes=nb_scripts)
        train_date_Y = np_utils.to_categorical(train_date_int,
                                               num_classes=nb_dates)

        model.fit({'img_input': train_X},
                  {'script_out': train_scr_Y,
                   'date_out': train_date_Y},
                  batch_size=BATCH_SIZE,
                  epochs=1,
                  shuffle=True)

        script_probas = []
        date_probas = []

        patch_script_gold = []
        patch_date_gold = []

        patch_script_pred = []
        patch_date_pred = []

        for dev_x, o_scr, o_date in zip(dev_crops, orig_dev_script_int,
                                        orig_dev_date_int):
            pred_scr_Y, pred_date_Y = \
                model.predict({'img_input': dev_x},
                              batch_size=BATCH_SIZE)

            patch_script_pred.extend(list(np.argmax(pred_scr_Y, axis=-1)))
            patch_date_pred.extend(list(np.argmax(pred_date_Y, axis=-1)))

            patch_script_gold.extend([o_scr] * NB_TEST_PATCHES)
            patch_date_gold.extend([o_date] * NB_TEST_PATCHES)

            av_pred_scr = pred_scr_Y.mean(axis=0)
            av_pred_date = pred_date_Y.mean(axis=0)

            script_probas.append(av_pred_scr)
            date_probas.append(av_pred_date)

        dev_script_pred = np.argmax(script_probas, axis=-1)
        dev_date_pred = np.argmax(date_probas, axis=-1)

        # patch level:
        curr_patch_script_acc = accuracy_score(patch_script_pred,
                                               patch_script_gold)
        print('  curr script val acc (patch level):', curr_patch_script_acc)

        curr_patch_date_acc = accuracy_score(patch_date_pred,
                                             patch_date_gold)
        print('  curr date val acc: (patch level)', curr_patch_date_acc)

        # image level:
        curr_script_acc = accuracy_score(dev_script_pred,
                                         orig_dev_script_int)
        print('  curr script val acc (image level):', curr_script_acc)

        curr_date_acc = accuracy_score(dev_date_pred,
                                       orig_dev_date_int)
        print('  curr date val acc:  (image level)', curr_date_acc)

        curr_acc = (curr_patch_script_acc + curr_patch_date_acc) / 2.0

        # save weights, if appropriate:
        if curr_acc > best_dev_acc:
            print('    -> saving model')
            model.save_weights('models/' + MODEL_NAME + '/weights.hdf5',
                               overwrite=True)
            best_dev_acc = curr_acc

        # half learning rate:
        if e and e % 20 == 0:
            old_lr = model.optimizer.lr.get_value()
            new_lr = np.float32(old_lr * 0.1)
            model.optimizer.lr.set_value(new_lr)
            print('\t- Lowering learning rate > was:', old_lr,
                  ', now:', new_lr)


if __name__ == '__main__':
    train()
