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

