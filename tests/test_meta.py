__author__ = 'headradio'

import hashlib

import numpy as np

import imagenet.dataset
import imagenet.dldatasets as dldatasets


def smoke_test():
    dataset = dldatasets.HvM_Categories()
    print dataset.meta[0]


def test_names():
    names = []
    dataset = imagenet.dldatasets.PixelHardSynsets20()
    names.append(dataset.specific_name)
    dataset = imagenet.dldatasets.PixelHardSynsets()
    names.append(dataset.specific_name)
    dataset = imagenet.dldatasets.HvM_Categories()
    names.append(dataset.specific_name)
    dataset = imagenet.dldatasets.Challenge_Synsets_100_Random()
    names.append(dataset.specific_name)
    dataset = imagenet.dldatasets.Challenge_Synsets_2_Pixel_Hard()
    names.append(dataset.specific_name)
    dataset = imagenet.dataset.Imagenet()
    names.append(dataset.specific_name)

    true_names = [
        'PixelHardSynsets_d052b5fa9f7026955c799ce238b096ac00615298',
        'PixelHardSynsets20_fb902a720218ad6cf75934c09f48f1678cc9c823',
        'PixelHardSynsets_d052b5fa9f7026955c799ce238b096ac00615298',
        'HvM_Categories_e5e3929ca9d206fa0f82767a0bf63c40bf7c586a',
        'Challenge_Synsets_100_Random_5048eda891e0a7ec15381684e593eecfa9dc9234',
        'Challenge_Synsets_100_Random_5048eda891e0a7ec15381684e593eecfa9dc9234',
        'Challenge_Synsets_2_Pixel_Hard_389367a954e9440e2ecf04aedd32684395600fa4',
        'Imagenet_6eef6648406c333a4035cd5e60d0bf2ecf2606d7']

    assert [x == y for x, y in zip(names, true_names)]
