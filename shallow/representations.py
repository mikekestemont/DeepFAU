import glob
from skimage import io
import pickle
import os

from skimage.transform import resize, downscale_local_mean
from scipy.spatial.distance import pdist, squareform
import numpy as np
from keras.models import model_from_json, Model
from distance_metrics import minmax

import utils
import pandas as pd


def main():
    MAX = None
    MODEL_NAME = 'SHALLOW'
    NB_ROWS, NB_COLS = 150, 150
    BATCH_SIZE = 30
    NB_TEST_PATCHES = 250

    folders = ['ICDAR2017_CLaMM_Training', 'ICDAR_CLAMM_task_1_3',
               'ICDAR_CLAMM_task_2_4']

    model = model_from_json(open('models/' + MODEL_NAME +
                            '/architecture.json').read())
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam', metrics=['accuracy'])
    model.load_weights('models/' + MODEL_NAME +
                       '/weights.hdf5')

    layer_name = 'global_average_pooling2d_1'
    get_repr = Model(inputs=model.input,
                     outputs=model.get_layer(layer_name).output)

    filenames, reprs = [], []
    for folder in folders:
        input_folder = '../data/' + folder
        for fn in glob.glob(input_folder+'/*'):
            if fn.endswith(('.db', '.csv')):
                continue
            print(fn)
            i = io.imread(fn, as_grey=True)
            img = np.array(i, dtype='float32')
            scaled = img / np.float32(255.0)
            scaled = downscale_local_mean(scaled, factors=(2, 2))

            X, _, _ = utils.augment_images(images=[scaled],
                                           scripts=[1],
                                           dates=[1],
                                           nb_rows=NB_ROWS,
                                           nb_cols=NB_COLS,
                                           nb_patches=NB_TEST_PATCHES,
                                           distort=False)
            R = get_repr.predict(X)
            for r in R:
                filenames.append(os.path.basename(fn))
                reprs.append(r)

    df = pd.DataFrame(reprs)
    df['filenames'] = filenames
    df.to_csv('../output/patch_representations.csv')


if __name__ == '__main__':
    main()
