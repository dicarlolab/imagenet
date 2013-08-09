imagenet
========

A python module containing both a full
imagenet dataset object conforming to skdata standards, and various related subsets.


For now, installing requires you have

This folder:
~/.skdata

Which is where the images will be cached locally

Under the hood, this initial version uses rsync to download files from a folder on mh17

This means you must configure your username to have passwordless ssh into mh17 for downloading images to work

Eventually this will be updated to use MongoDB on dicarloX



To install:
===============

pip install git+http://github.com/dicarlolab/imagenet.git#egg=imagenet

or if you don't have root access

pip install --user -e git+http://github.com/dicarlolab/imagenet.git#egg=imagenet


Some examples:
=====================


import imagenet.dataset as d

dataset = d.HvM_Categories_Approximated_by_Synsets()
translation_dictionary = dataset.translation_dict  # Dictionary showing the mapping between hvm categories and synsets that I used
meta = dataset.meta
images = dataset.get_images()   # uses the dataset.default_preproc spec, which is a property of the dataset. we can change it for now, but I think in the future if we want to use different preprocs, we should extend the appropriate class and set its default_preproc property in the init method
