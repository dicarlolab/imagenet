__author__ = 'headradio'

import hashlib

import numpy as np

import imagenet.dataset
import imagenet.dldatasets as dldatasets
import random


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

    assert all([x == y for x, y in zip(names, true_names)])


def test_image_source():
    dataset = imagenet.dataset.Imagenet()
    filenames = dataset.meta['filename']
    source = imagenet.dataset.get_img_source()
    hashes = [imagenet.dataset.get_id(source.get(file).read()) for file in random.sample(filenames, 10)]
    true_hashes = [
        'd1d2bf593b3a4d5bd591dbdb6d96c5942a7e1bd2',
        '4684789e1e1dab4c0ccdfd76967c44e16e66f1b6',
        '0c8c452b7a3449656b4051dcda4687627c8c0d35',
        'a06e86c76e0c9fcd79be4876cbb35a968d834979',
        '9d489315a49e223c35443d502e794dd82c7109f7',
        'ba52edbddb3d96d108cd918bb36d6b11f50b8a47',
        '805ae093c388e93888fd20e04bcf048d380e19dc',
        '1c56afe3fdec6c41fba94e4672af5840261fd351',
        '2ea2efa505c321a33406747ecc96aee16f8a4931',
        '37756c374b1b2bf8f81cde0e7e48ff021f48c749']
    assert all([x == y for x, y in zip(hashes, true_hashes)])
