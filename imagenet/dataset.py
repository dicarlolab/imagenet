"""A module docstring that describes the nature of the data set, the web site that describes the data set more fully,
0;95;cand contain relevant references to academic literature.

Logic for downloading the data set from the most official internet distribution location possible.
Logic for unpacking and loading that data set into primitive Python data types, if possible."""

import os
import tarfile
import cPickle
import itertools
from urllib2 import urlopen
from collections import defaultdict
import networkx
import numpy as np
import tabular as tb
import skdata.larray as larray
from skdata.data_home import get_data_home
from bs4 import BeautifulSoup
import random
from random import sample
from dldata.stimulus_sets.dataset_templates import get_id
from dldata.stimulus_sets import dataset_templates
from joblib import Parallel, delayed
import pymongo as pm
import gridfs

try:
    from nltk.corpus import wordnet as wn
except ImportError:
    wn = None
    print 'You must download wordnet using nltk.download() (see readme)'
    raise ValueError
#These synsets are reported by the api, but cannot be downloaded 9/16/2013
broken_synsets = {'n04399382'}


def descendants(graph, source):
    return set(networkx.shortest_path_length(graph, source).keys()) - {source}

#TODO : deal with username and accesskey so that we can share this code
IMAGENET_DB_PORT = int(os.environ.get('IMAGENET_DB_PORT', 27017))
IMAGENET_SOURCE_DB = pm.MongoClient('localhost', port=IMAGENET_DB_PORT).gridfs_example
IMAGENET_FS = gridfs.GridFS(IMAGENET_SOURCE_DB)


def get_img_source():
    return IMAGENET_FS


def download_images_by_synset(synsets, seed=None, num_per_synset='all', firstonly=False, path=None,
                              imagenet_username='ardila', accesskey='bd662acb4866553500f17babd5992810e0b5a439'):
    """
    Stores a random #num images for synsets specified by synsets from the latest release to path specified
    Since files are stored as tar files online, the entire synset must be downloaded to access random images.

    If 'all' is passed as the num argument, all images are stored.

    If the argument firstonly is set to true, then download times can be reduced by only extracting the first
    few images

    Returns a meta tabarray object containing wnid and filename for each downloaded image
    """
    if path is None:
        path = os.getcwd()
    if not os.path.exists(path):
        os.makedirs(path)
    synsets = list(synsets)
    random.seed(seed)
    kept_names = []
    kept_synset_list = []
    for i, synset in enumerate(synsets):
        synset_names = []
        url = 'http://www.image-net.org/download/synset?' + \
              'wnid=' + str(synset) + \
              '&username=' + imagenet_username + \
              '&accesskey=' + accesskey + \
              '&release=latest'
        print i
        print url
        url_file = urlopen(url)
        tar_file = tarfile.open(fileobj=url_file, mode='r|')
        if firstonly and not (num_per_synset == 'all'):
            keep_idx = xrange(num_per_synset)
            for tarinfo in tar_file:
                synset_names.append(tarinfo.name)
                tar_file.extract(tarinfo, path)
        else:
            for tarinfo in tar_file:
                synset_names.append(tarinfo.name)
                tar_file.extract(tarinfo, path)
            if num_per_synset == 'all':
                keep_idx = range(len(synset_names))
            else:
                keep_idx = sample(range(len(synset_names)), num_per_synset)
            files_to_remove = frozenset(synset_names) - frozenset([synset_names[idx] for idx in keep_idx])
            for file_to_remove in files_to_remove:
                os.remove(path + '/' + file_to_remove)
        kept_names.extend([synset_names[idx] for idx in keep_idx])
        kept_synset_list.extend([synset] * len(keep_idx))
    meta = tb.tabarray(records=zip(kept_names, kept_synset_list), names=['filename', 'synset'])
    return meta


def update_gridfs_with_synsets(synsets, fs, force=True,
                               imagenet_username='ardila', accesskey='bd662acb4866553500f17babd5992810e0b5a439'):
    filenames_dict = cPickle.loads(fs.get('filenames_dict.p').read())
    for i, synset in enumerate(synsets):
        filenames = []
        url = 'http://www.image-net.org/download/synset?' + \
              'wnid=' + str(synset) + \
              '&username=' + imagenet_username + \
              '&accesskey=' + accesskey + \
              '&release=latest'
        print i
        print url
        url_file = urlopen(url)
        tar_file = tarfile.open(fileobj=url_file, mode='r|')

        for tar_info in tar_file:
            filename = tar_info.name
            not_uploaded = True
            while not_uploaded:
                try:
                    if force:
                        fs.delete(filename)
                        print 'Overwriting ' + filename
                    filenames.append(filename)
                    fs.put(tar_file.extractfile(tar_info), _id=filename)
                    not_uploaded = False
                except IOError:
                    print filename + ' Failed'
        filenames_dict[synset] = filenames
    fs.delete('filenames_dict.p')
    file_obj = open(os.path.join(get_data_home(), 'imagenet', 'filenames_dict.p'), 'wb')
    cPickle.dump(filenames_dict, file_obj)
    file_obj.close()
    file_obj = open(os.path.join(get_data_home(), 'imagenet', 'filenames_dict.p'), 'rb')
    fs.put(file_obj, _id='filenames_dict.p')


def download_2013_ILSCRV_synsets(num_per_synset='all', seed=None, path=None, firstonly=False):
    """
    Stores a random #num images for the 2013 ILSCRV synsets from the latest release.
    Since files are stored as tar files online, the entire synset must be downloaded to access random images.

    If 'all' is passed as the num argument, all images are stored.

    If the argument firstonly is set to true, then download times can be reduced by only extracting the first
    few images

    Returns a tabular meta object that has a record for each image containing 2 fields
        synset: the synset of the image
        filename: the filename of the image
    """
    synsets = get2013_Categories()
    synsets = set(synsets) - broken_synsets
    return download_images_by_synset(synsets, seed=seed, num_per_synset=num_per_synset, path=path, firstonly=firstonly)


def get2013_Categories():
    """Get list of synsets in 2013 ILSCRV Challenge by scraping the challenge's website"""
    name_list = []
    synset_list = []
    #Grabbed website to extract synsets for all images
    parser = BeautifulSoup(urlopen("http://www.image-net.org/challenges/LSVRC/2013/browse-synsets"))

    def is_a_2013_category(tag):
        """
            Returns true if the tag is a link to a category in the 2013 challenge
            :type tag: tag
            :rtype : boolean
            :param tag: tag object
            """
        if tag.has_attr('href'):
            if 'synset' in tag['href']:
                return True
        else:
            return False

    linkTags = parser.findAll(is_a_2013_category)
    for linkTag in linkTags:
        name_list.append(linkTag.string)
        link = linkTag['href']
        synset_list.append(link.partition('=')[2])
    return synset_list


def save_filename_dict_from_img_folder(path=None):
    """
    Run this code at IMG_SOURCE to build the dictionary.
    os.listdir is very slow, so allow for about 24hr runtime for large img folders
    """
    if path is None:
        path = os.getcwd()
    filenames_dict = defaultdict(list)
    filenames = os.listdir(path)
    imgs = [f for f in filenames if f.endswith('.JPEG')]
    for f in imgs:
        synset = f.split('_')[0]
        # im_id = f.split('_')[1].rstrip('.JPEG')
        filenames_dict[synset].append(f)
    cPickle.dump(filenames_dict, open('filenames_dict.p', 'wb'))


def get_word_dictionary():
    words_text = urlopen("http://www.image-net.org/archive/words.txt").readlines()
    word_dictionary = {}
    for row in words_text:
        word_dictionary[row.split()[0]] = ' '.join(row.split()[1:]).rstrip('\n')
    return word_dictionary


def get_definition_dictionary():
    gloss_text = urlopen("http://www.image-net.org/archive/gloss.txt").readlines()
    definition_dictionary = {}
    for row in gloss_text:
        definition_dictionary[row.split()[0]] = ' '.join(row.split()[1:]).rstrip('\n')
    return definition_dictionary


class Imagenet_Base(dataset_templates.ImageDatasetBase):
    def __init__(self, data=None):

        """

        :param data: data specifying how to build this dataset. should uniquely identify dataset among all datasets
        :raise: ValueError if instantiated directly
        """
        cname = self.__class__.__name__
        if cname == 'Imagenet_Base':
            print 'The Imagenet base class should not be directly instantiated'
            raise ValueError

        img_path = self.imagenet_home('images')
        self.specific_name = self.__class__.__name__ + '_' + get_id(data)
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        self.img_path = img_path

        self.meta_path = self.local_home('meta')
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        self.default_preproc = {'resize_to': (256, 256), 'mode': 'RGB', 'dtype': 'float32',
                                'crop': None, 'mask': None, 'normalize': True}
        super(Imagenet_Base, self).__init__(data)

    def imagenet_home(self, *suffix_paths):
        return os.path.join(get_data_home(), 'imagenet', *suffix_paths)

    def local_home(self, *suffix_paths):
        return self.imagenet_home(self.specific_name, *suffix_paths)

    def home(self, *suffix_paths):
        return self.local_home(*suffix_paths)

    def fetch(self):
        pass

    @property
    def filenames(self):
        return self.meta['filename']

    def get_meta(self):
        """Loads the synset meta from file, if it exists.
        If it doesn't exist, calls _get_meta"""
        try:
            tabular_load = tb.io.loadbinary(os.path.join(self.meta_path, 'meta.npz'))
            # This seems like a flaw with tabular's loadbinary.
            meta = tb.tabarray(records=tabular_load[0], dtype=tabular_load[1])
        except IOError:
            print 'Could not load meta from file, constructing'
            s = self.synset_meta
            filenames = list(itertools.chain.from_iterable(
                [s[synset]['filenames'] for synset in self.synset_meta.keys()]))
            synsets = [filename.split('_')[0] for filename in filenames]
            meta = tb.tabarray(records=zip(filenames, synsets, filenames), names=['filename', 'synset', 'id'])
            tb.io.savebinary(os.path.join(self.meta_path, 'meta.npz'), meta)
        return meta

    @property
    def synset_meta(self):
        if not hasattr(self, '_synset_meta'):
            self._synset_meta = self._get_synset_meta()
        return self._synset_meta

    @property
    def full_tree_structure(self):
        if not hasattr(self, '_full_tree_structure'):
            self._full_tree_structure = self._get_full_tree_structure()
        return self._full_tree_structure

    def _get_full_tree_structure(self):
        filename = 'full_tree_structure_wordnet.p'
        folder = self.imagenet_home()
        try:
            full_tree_structure = cPickle.load(open(os.path.join(folder, filename), 'rb'))
        except IOError:
            print "Calculating full tree structure using wordnet"
            imagenet_format = lambda synset: 'n' + "%08d" % synset.offset
            full_tree_structure = networkx.DiGraph()
            [[full_tree_structure.add_edge(imagenet_format(parent), imagenet_format(child))
              for child in parent.hyponyms()] for parent in wn.all_synsets()]
            cPickle.dump(full_tree_structure, open(os.path.join(folder, filename), 'wb'))
            print 'done'
        return full_tree_structure

    def _get_synset_meta(self):
        """Loads the synset meta from file, if it exists.
        If it doesn't exist, calls _get_synset_meta"""
        try:
            synset_meta = cPickle.load(open(os.path.join(self.meta_path, 'synset_meta.p'), 'rb'))
        except IOError:
            print 'Could not load synset meta from file, constructing'
            self.synset_list = self.get_synset_list()
            words = get_word_dictionary()
            definitions = get_definition_dictionary()
            filenames = self.get_filename_dictionary()
            synset_meta = dict([(synset, {'words': words[synset],
                                          'definition': definitions[synset],
                                          'filenames': filenames[synset],
                                          'num_images': len(filenames[synset])})
                                for synset in self.synset_list])
            cPickle.dump(synset_meta, open(os.path.join(self.meta_path, 'synset_meta.p'), 'wb'))
        return synset_meta

    def get_synset_list(self, thresh=0):
        """
        thresh: int, minimum number of files to be included on the list
        """
        if not hasattr(self, 'synset_list'):
            all_synsets_url = 'http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list'
            self.synset_list = [wnid.rstrip() for wnid in urlopen(all_synsets_url).readlines()[:-2]]
        rval = self.synset_list
        if thresh > 0:
            rval = filter(lambda x: self.synset_meta[x]['num_images'] >= thresh, rval)
        return rval

    def overlapping_tuples(self, synset_list=None):
        tree_struct = self.full_tree_structure
        if synset_list is None:
            synset_list = self.get_synset_list()
        overlapping_tuples = []
        for s1, s2 in itertools.combinations(synset_list, 2):
            if s2 in descendants(tree_struct, s1):
                overlapping_tuples.append((s1, s2))
        return overlapping_tuples

    def get_full_filenames_dictionary(self):
    #This is a (maybe _the_) key piece of metadata, so it is installed to a specific location locally
        filename = 'filenames_dict.p'
        folder = self.imagenet_home()
        try:
            filenames_dict = cPickle.load(open(os.path.join(folder, filename), 'rb'))
        except IOError:
            print 'Filename dictionary not found, attempting to copy from IMG_SOURCE'
            download_file_to_folder(filename, folder)
            filenames_dict = cPickle.load(open(os.path.join(folder, filename), 'rb'))
        return filenames_dict

    def get_filename_dictionary(self):
        filename = os.path.join(self.meta_path, 'filenames_dict.p')
        if not hasattr(self, 'filenames_dict'):
            try:
                self.filenames_dict = cPickle.load(open(filename, 'rb'))
            except IOError:
                print 'Could not load filenames_dict from file, constructing from full filenames_dict'
                full_dict = self.get_full_filenames_dictionary()
                synset_list = self.get_synset_list()
                self.filenames_dict = {synset: full_dict[synset] for synset in synset_list}
                cPickle.dump(self.filenames_dict, open(filename, 'wb'))
        return self.filenames_dict

    def get_images(self, preproc, n_jobs=-1, cache=False):
        """
        Create a lazily reevaluated array with preprocessing specified by a preprocessing dictionary
        preproc. See the documentation in ImgDownloaderCacherPreprocesser

        """
        file_names = self.meta['filename']
        img_source = get_img_source()
        cachedir = self.imagenet_home('images')
        processor = ImgDownloaderPreprocessor(
            source=img_source, preproc=preproc, n_jobs=n_jobs, cache=cache, cachedir=cachedir)
        return larray.lmap(processor,
                           file_names,
                           f_map=processor)

    def get_pixel_features(self, preproc=None, n_jobs=-1):
        preproc['flatten'] = True
        return self.get_images(preproc, n_jobs)


def download_file_to_folder(filename, folder, source=get_img_source(), verbose=False):
    if verbose:
        print filename
    file_from_source = source.get(filename)
    file_obj = open(os.path.join(folder, filename), 'w')
    file_obj.write(file_from_source.read())
    file_obj.close()


class ImgDownloaderPreprocessor(dataset_templates.ImageLoaderPreprocesser):
    """
    Class used to lazily downloading images to a cache, evaluating resizing/other pre-processing
    and loading image from file in an larray
    """

    def __init__(self, source, preproc, n_jobs=-1, cache=False, cachedir=None):
        """
        :param source: string, adress passable to rsync where images are located
        :param preproc: A preprocessing spec. A preprocessing spec is a dictionary containing:
            resize_to: Image is resized to the tuple given here (note: not reshaped)
            dtype: The datatype of the image array
            mode: 'RGB' or 'L' sepcifies whether or not to store color images
            mask: Image object which is used to mask the image
            crop: array of [minx, maxx, miny, maxy] crop box applied after resize
            normalize: If true, then the image set to zero mean and unit standard deviation

        """
        self.source = source
        self.preproc = preproc
        self.n_jobs = n_jobs
        self.cache = cache
        self.cachedir = cachedir
        super(ImgDownloaderPreprocessor, self).__init__(preproc)

    def __call__(self, file_names):
        """
        :param file_names: file_names to download and preprocess
        :return: image
        """
        if isinstance(file_names, str):
            file_names = np.array([file_names])
        blocksize = 1
        numblocks = int(math.ceil(len(file_names) / float(blocksize)))
        filename_blocks = [file_names[i * blocksize: (i + 1) * blocksize].tolist() for i in range(numblocks)]
        results = Parallel(
            n_jobs=self.n_jobs, verbose=100)(
            delayed(download_and_process)(filename_block, self.preproc, cache=self.cache, cachedir=self.cachedir)
            for filename_block in filename_blocks)
        results = list(itertools.chain(*results))
        if len(file_names) > 1:
            return np.asarray(results)
        else:
            return np.asarray(results)[0]
            # return np.asarray(map(self.load_and_process, np.asarray(file_paths)))


import math


def download_and_process(file_names, preproc, cache=False, cachedir=None):
    processer = dataset_templates.ImageLoaderPreprocesser(preproc)
    rvals = [download_and_process_core(fname, processer, cache, cachedir) for fname in file_names]
    return rvals


def download_and_process_core(file_name, processer, cache, cachedir):
    """
    :param file_name: which file to download
    :param processer: preprocesser object to use (see ImageLoaderPreprocesser)
    :return: array of preprocessed image
    """
    fs = get_img_source()
    if cache:
        path = os.path.join(cachedir, file_name)
        if not os.path.isfile(path):
            print('Downloading and caching: %s' % path)
            fileobj = fs.get(file_name)
            with open(path, 'wb') as _f:
                _f.write(fileobj.read())
        fileobj = open(path)
    else:
        fileobj = fs.get(file_name)
        # file_like_obj = cStringIO(grid_file.read())
    try:
        rval = processer.load_and_process(fileobj)
    except IOError:
        print 'Image ' + file_name + 'is broken, will be replaced with zeros'
        # if os.path.exists('broken_images.p'):
        #     broken_list = cPickle.load(open('broken_images.p', 'rb')).append(file_name)
        # else:
        #     broken_list = []
        # cPickle.dump(broken_list, open('broken_images.p', 'wb'))
        rval = np.zeros(processer.load_and_process(fs.get('n04135315_18202.JPEG')).shape)
    return rval


class Imagenet_synset_subset(Imagenet_Base):
    def __init__(self, data):
        """
        data has 1 field you can setparams
            synset_list: List of synsets to include in this subset
        """
        self.synset_list = data['synset_list']
        super(Imagenet_synset_subset, self).__init__(data=data)


class Imagenet_filename_subset(Imagenet_synset_subset):
    def __init__(self, data):
        self.filenames_dict = data.get('filenames_dict')
        filenames = data['filenames']
        if self.filenames_dict is None:
            self.filenames_dict = defaultdict(list)
            synset_list = []
            for f in filenames:
                synset = f.split('_')[0]
                self.filenames_dict[synset].append(f)
                if synset not in synset_list:
                    synset_list.append(synset)
        data['synset_list'] = self.filenames_dict.keys()
        super(Imagenet_filename_subset, self).__init__(data=data)


class Imagenet(Imagenet_Base):
    """All the images in Imagenet.
    """
    pass
