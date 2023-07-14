"""
Module related to the click-me dataset
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob 
from .utils import get_synset

from .blur import gaussian_kernel, gaussian_blur

CLICKME_BASE_URL = 'https://storage.googleapis.com/serrelab/prj_harmonization/dataset/click-me'
PSYCH_BASE_URL = 'https://storage.googleapis.com/serrelab/prj_harmonization/dataset/psychophysics/clicktionary'

NB_VAL_SHARDS = 17
NB_TRAIN_SHARDS = 318

AUTO = tf.data.AUTOTUNE
GAUSSIAN_KERNEL = tf.cast(gaussian_kernel(), tf.float32)
FEATURE_DESCRIPTION = {
      "image"       : tf.io.FixedLenFeature([], tf.string, default_value=''),
      "heatmap"     : tf.io.FixedLenFeature([], tf.string, default_value=''),
      "label"       : tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


def parse_clickme_prototype(prototype):
    """
    Parses a Click-me prototype.

    Parameters
    ----------
    prototype : tf.Tensor
        The Click-me prototype to parse.

    Returns
    -------
    image : tf.Tensor
        The image.
    heatmap : tf.Tensor
        The heatmap.
    label : tf.Tensor
        The label.
    """
    # parse a single sample
    data = tf.io.parse_single_example(prototype, FEATURE_DESCRIPTION)

    # load & preprocess image
    image   = tf.io.decode_jpeg(data['image'])
    image   = tf.reshape(image, (256, 256, 3))
    image   = tf.cast(image, tf.float32)
    image   = tf.image.resize(image, (224, 224), method='bilinear')

    # load & blur the heatmap
    heatmap = tf.io.decode_jpeg(data['heatmap'])
    heatmap = tf.reshape(heatmap, (256, 256, 1))
    heatmap = tf.cast(heatmap, tf.float32)
    heatmap = tf.image.resize(heatmap, (64, 64), method="bilinear")
    heatmap = gaussian_blur(heatmap, GAUSSIAN_KERNEL)
    heatmap = tf.image.resize(heatmap, (224, 224), method="bilinear")

    label   = tf.cast(data['label'], tf.int32)
    label   = tf.one_hot(label, 1_000)

    return image, heatmap, label


def load_clickme(shards_paths, batch_size):
    """
    Loads the click-me dataset (training or validation).

    Parameters
    ----------
    shards_paths : list of str
        The path to the shards to load.
    batch_size : int, optional
        Batch size, by default 64

    Returns
    -------
    dataset
        A `tf.dataset` of the Click-me dataset.
        Each element contains a batch of (images, heatmaps, labels).
    """
    deterministic_order = tf.data.Options()
    deterministic_order.experimental_deterministic = True

    dataset = tf.data.TFRecordDataset(shards_paths, num_parallel_reads=AUTO)
    dataset = dataset.with_options(deterministic_order)

    dataset = dataset.map(parse_clickme_prototype, num_parallel_calls=AUTO)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)

    return dataset


def load_clickme_train(batch_size = 64):
    """
    Loads the click-me training set.

    Parameters
    ----------
    batch_size : int, optional
        Batch size, by default 64

    Returns
    -------
    dataset
        A `tf.dataset` of the Click-me training dataset.
        Each element contains a batch of (images, heatmaps, labels).
    """

    shards_paths = [
        tf.keras.utils.get_file(f"clickme_train_{i}",
                                f"{CLICKME_BASE_URL}/train/train-{i}.tfrecords",
                               cache_subdir="datasets/click-me") for i in range(NB_TRAIN_SHARDS)
    ]

    return load_clickme(shards_paths, batch_size)


def load_clickme_val(batch_size = 64):
    """
    Loads the click-me validation set.

    Parameters
    ----------
    batch_size : int, optional
        Batch size, by default 64

    Returns
    -------
    dataset
        A `tf.dataset` of the Click-me validation dataset.
        Each element contains a batch of (images, heatmaps, labels).
    """

    shards_paths = [
        tf.keras.utils.get_file(f"clickme_val_{i}",
                                f"{CLICKME_BASE_URL}/val/val-{i}.tfrecords",
                                cache_subdir="datasets/click-me") for i in range(NB_VAL_SHARDS)
    ]

    return load_clickme(shards_paths, batch_size)

def get_human_data(stimuli_folder):
    """
    Loads the human data from the click-me dataset.

    Parameters
    ----------
    stimuli_folder : str
        Path to the stimuli folder.


    Returns
    -------
    ims : list
        List of images.
    human_data : list
        List of human data.
    column_names : list 
        List of column names.


    """ 
    data = np.load(os.path.join(stimuli_folder,"data_for_zahra.npz"), allow_pickle=True, encoding="latin1")

    ims = data["ims"]
    human_data = data["data_human"]
    column_names = data["columns"]

    im_cats = [x.split(os.path.sep)[1][:-1] for x in np.concatenate(ims)]
    unique_categories = np.unique(im_cats)

    df = pd.DataFrame(human_data, columns=column_names.astype(str))
    mean_perfs = df.groupby("Revelation").mean().reset_index()
    std_perfs = df.groupby("Revelation").std().reset_index()
    exp_perfs = df[df.Revelation < 200.]
    exp_perf_means = mean_perfs[:-1]
    full_perf = df[df.Revelation == 200.]
    exp_perfs['correct'] = exp_perfs['correct']/exp_perfs['correct'].max()

    mpx = mean_perfs.iloc[:-1]["Revelation"]
    mpy = mean_perfs.iloc[:-1]["correct"]
    mpz = std_perfs.iloc[:-1]["correct"]

    mpy =(mpy -np.min(mpy))/(mpy.max()-np.min(mpy))

    mpx = mpx.tolist()
    mpy= mpy.tolist()
    mpx = mpx[:-1]+mpx[-1:]
    mpy = mpy[:-1]+mpy[-1:]
    return mpx, mpy , mpz

def get_stimuli_paths(stimuli_folder = None):
    """
    Returns the paths to the stimuli.

    Parameters
    ----------
    stimuli_folder : str, optional
        Path to the stimuli folder, by default None

    Returns
    -------
    list    
        List of paths to the stimuli.
    """
    p1 = os.path.join(stimuli_folder,'exp_1_clicktionary_probabilistic_region_growth_centered')
    p2 = os.path.join(stimuli_folder,'exp_2_clicktionary_probabilistic_region_growth_centered')
    images1 = glob.glob(os.path.join(p1,'*png')) 
    images2 = glob.glob(os.path.join(p2,'*png'))
    return images1+images2



def load_psychophysics():
    """
    Loads the psychophysics dataset.


    Parameters
    ----------
    batch_size : int, optional
        Batch size, by default 64

    Returns
    -------
    dataset
        A `tf.dataset` of the psychophysics dataset.
        Each element contains a batch of (images, heatmaps, labels).
    """
    _,_,revmap = get_synset()
    dataset=[]
    stimuli = get_stimuli_paths()
    for im in stimuli:
        file = im.split('/')[-1]
        label = ''.join(file[:-5].split('_')[1:])
        indx_label = revmap[label]['index']
        if indx_label <398: 
            task_label = 1
        else: 
            task_label = 0
        diff = file.split('_')[0]
        sample = file[-5]
        dataset.append([im,file,label,diff,sample,indx_label,task_label])
    exp_df = pd.DataFrame(dataset,columns=['path','name','label','difficulty','sample number','imagenet_index_label','task_label'])
    return exp_df

def get_psychophysics():
    """
    Loads the psychophysics dataset.

    Returns
    -------
    dataset
    """
    
    folder = tf.keras.utils.get_file("psychophysics",PSYCH_BASE_URL,cache_subdir="datasets/psychophysics")
    mpx, mpy , mpz = get_human_data(folder)
    exp_df = load_psychophysics()
    stimuli_paths = get_stimuli_paths(folder)
    return mpx, mpy , mpz, stimuli_paths,exp_df
  