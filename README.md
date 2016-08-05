imagenet
========

A python module containing both a full
imagenet dataset object conforming to skdata standards, and various related subsets.

This folder will be created when using datasets:
~/.skdata/imagenet/images

Which is where the images will may be cached locally (not default, see dataset.get_images for documentation)

You will need a user account on our database (or a similarly configured one). Email ardila@mit.edu.


To install:
===============

```
$ pip install git+http://github.com/dicarlolab/imagenet.git#egg=imagenet
```

or if you don't have root access

```
$ pip install --user -e git+http://github.com/dicarlolab/imagenet.git#egg=imagenet
```
you have to install the requirements from the requirements file as well

```
pip install -r requirements.txt
```

tunnel to the database

```
ssh -f -N -L 27017:localhost:27017 username@dicarlo5.mit.edu
```

you must also use the nltk package to download wordnet

``` python
nltk.download('wordnet')
```

Some examples:
=====================

Import the dataset and call its constructor


```
import imagenet.dldatasets as d
dataset = d.Challenge_Synsets_20_Pixel_Hard()
```


The dataset has a meta tabular array object

```
meta = dataset.meta
```

And a dictionary containing a dictionary of information about each synset, each of which is represented by a wordnet id

```
synset_meta = dataset.synset_meta
list_of_wordnet_ids = synset_meta.keys()
info_about_first_synset = synset_meta[list_of_wordnet_ids[0]].keys()
```

get_images() can use the dataset.default_preproc spec

```
images = dataset.get_images(dataset.default_preproc)
```






