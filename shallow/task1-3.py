import glob
from skimage import io
import pickle
import os

from skimage.transform import resize, downscale_local_mean
from scipy.spatial.distance import pdist, squareform
import numpy as np
from keras.models import model_from_json
from distance_metrics import minmax

import utils


def main():
    MAX = None
    MODEL_NAME = 'SHALLOW'
    NB_ROWS, NB_COLS = 150, 150
    BATCH_SIZE = 30
    NB_TEST_PATCHES = 50

    images, filenames = [], []
    for fn in glob.glob('../data/ICDAR_CLAMM_task_1_3/*.tif'):
        img = np.array(io.imread(fn, as_grey=True), dtype='float32')
        scaled = img / np.float32(255.0)
        scaled = downscale_local_mean(scaled, factors=(2, 2))
        images.append(scaled)
        filenames.append(os.path.basename(fn))
        if MAX and len(images) >= MAX:
            break

    script_encoder = \
        pickle.load(open('models/' +
                    MODEL_NAME + '/script_encoder.p', 'rb'))
    date_encoder = \
        pickle.load(open('models/' +
                    MODEL_NAME + '/date_encoder.p', 'rb'))

    model = model_from_json(open('models/' + MODEL_NAME +
                            '/architecture.json').read())
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam', metrics=['accuracy'])
    model.load_weights('models/' + MODEL_NAME +
                       '/weights.hdf5')

    script_probas, date_probas = [], []
    for image, filename in zip(images, filenames):
        print(filename)
        X, _, _ = utils.augment_images(images=[image],
                                       scripts=[1],
                                       dates=[1],
                                       nb_rows=NB_ROWS,
                                       nb_cols=NB_COLS,
                                       nb_patches=NB_TEST_PATCHES,
                                       distort=False)
        pred_scr_Y, pred_date_Y = \
            model.predict({'img_input': X},
                          batch_size=BATCH_SIZE)

        pred_scr = np.array(pred_scr_Y)
        pred_date = np.array(pred_date_Y)

        av_pred_scr = pred_scr.mean(axis=0)
        av_pred_date = pred_date.mean(axis=0)

        script_probas.append(av_pred_scr)
        date_probas.append(av_pred_date)

    # Task 1: script for clean data (13 cols)
    labels = [str(i) for i in range(1, 13)]
    idxs = [list(script_encoder.classes_).index(l) for l in labels]
    with open('../output/task1_clean_script_belonging.txt',
              'w') as f:
        for fn, probas in zip(filenames, script_probas):
            f.write(fn + ', ')
            scores = [str(probas[i]) for i in idxs]
            f.write(', '.join(scores))
            f.write('\n')

    # Task 3: date for clean data (13 cols)
    labels = [str(i) for i in range(1, 16)]
    idxs = [list(date_encoder.classes_).index(l) for l in labels]
    with open('../output/task3_clean_date_belonging.txt',
              'w') as f:
        for fn, probas in zip(filenames, date_probas):
            f.write(fn + ', ')
            scores = [str(probas[i]) for i in idxs]
            f.write(', '.join(scores))
            f.write('\n')

    # Task 1: distances for clean data (13 cols)
    X = np.asarray(script_probas)
    dm = squareform(pdist(X, minmax))
    dm = dm / dm.sum(axis=1)[:, np.newaxis]
    with open('../output/task1_clean_script_distance.txt',
              'w') as f:
        f.write('L/C ,')
        f.write(', '.join(filenames)+'\n')
        for n, v in zip(filenames, dm):
            f.write(n + ', ')
            f.write(', '.join([str(sc) for sc in v]))
            f.write('\n')

    # Task 3: distances for clean data (13 cols)
    X = np.asarray(date_probas)
    dm = squareform(pdist(X, minmax))
    dm = dm / dm.sum(axis=1)[:, np.newaxis]
    with open('../output/task3_clean_date_distance.txt',
              'w') as f:
        f.write('L/C ,')
        f.write(', '.join(filenames)+'\n')
        for n, v in zip(filenames, dm):
            f.write(n + ', ')
            f.write(', '.join([str(sc) for sc in v]))
            f.write('\n')


if __name__ == '__main__':
    main()
