import itertools
import random
import numpy as np
from dataset import (Imagenet_filename_subset, 
                     Imagenet_synset_subset,
                     get2013_Categories,
                     Imagenet)


class HvM_Categories(Imagenet_filename_subset):
    """
    Hand-chosen imagenet equivalents of HvM categories
    Has an attribute called translation dict explaining the mapping
    """
    def __init__(self):
        full_dict = self.get_full_filename_dictionary()
        self.translation_dict = \
            {'Animals': 'n00015388',
             'Boats': 'n02858304',
             'Cars': 'n02958343',
             'Chairs': 'n03001627',
             'Faces': 'n09618957',
             'Fruits': 'n13134947',
             'Planes': 'n02691156',
             'Tables': 'n04379243'}

        synset_list = self.translation_dict.values()
        #8000 might still be too many images, here I'm subsetting
        # the synsets to get a size similar to one of the variation levels
        filenames = list(itertools.chain.from_iterable(full_dict[synset][0:200] for synset in synset_list))
        self._old_filenames = list(itertools.chain.from_iterable(full_dict[synset][0:200] for synset in synset_list))
        data = {'filenames': filenames}
        super(HvM_Categories, self).__init__(data=data)

    def memmap_pixel_features(self):
        import os.path as path
        filename = path.join(self.img_path, 'HvM.dat')
        return np.memmap(filename, 'float32', 'w+')


#this is the dataset we used in pixel screening
class Challenge_Synsets_100_Random(Imagenet_filename_subset):
    def __init__(self):
        random.seed('Challenge_Synsets_100_Random')
        synsets = random.sample(get2013_Categories(), 100)
        num_per_synset = 200
        full_dict = self.get_full_filename_dictionary()
        filenames = []
        for synset in synsets:
            filenames.extend(random.sample(full_dict[synset], num_per_synset))
        data = {'filenames': filenames}
        super(Challenge_Synsets_100_Random, self).__init__(data=data)


class Challenge_Synsets_20_Pixel_Hard(Imagenet_synset_subset):
    def __init__(self):
        synsets =   \
            ['n03196217',
             'n02236044',
             'n02091831',
             'n03259280',
             'n02102318',
             'n02795169',
             'n03633091',
             'n03950228',
             'n03804744',
             'n03424325',
             'n04265275',
             'n04200800',
             'n03666591',
             'n02097474',
             'n02096177',
             'n04507155',
             'n02109047',
             'n02093428',
             'n04599235',
             'n04153751']
        data = {'synset_list': synsets}
        super(Challenge_Synsets_20_Pixel_Hard, self).__init__(data=data)


class Challenge_Synsets_2_Pixel_Hard(Imagenet_synset_subset):
    def __init__(self):
        synsets =   \
            ['n03196217',
             'n02236044']
        data = {'synset_list': synsets}
        super(Challenge_Synsets_2_Pixel_Hard, self).__init__(data=data)


class Big_Pixel_Screen(Imagenet_filename_subset):
    def __init__(self):
        random.seed('Big_Pixel_Screen')
        big_synsets = Imagenet().get_synset_list(thresh=1000)
        screen_synsets = list(set(get2013_Categories()) | big_synsets)
        filename_dict = self.get_full_filename_dictionary()
        filenames = []
        num_per_synset = 200
        for synset in screen_synsets:
            filenames.extend(random.sample(filename_dict[synset], num_per_synset))
        data = {'filenames': filenames}
        super(Big_Pixel_Screen, self).__init__(data=data)

# import tarfile
# import os
# f = open(os.path.expanduser('~/output.txt')).readlines()
# f = [a.rstrip() for a in f]
# tars = [a for a in f if a.endswith('.tar')]
# jpegs = set([a for a in f if a.endswith('.JPEG')])
# file_list = []
# tars = [synset+'.tar' for synset in get2013_Categories()]
# total = float(len(tars))
# for i, tar in enumerate(tars):
#     print tar
#     print i/total
#     tar_file = tarfile.open(os.path.expanduser('~/imagenet_download/')+tar)
#     tar_file.extractall(os.path.expanduser('~/imagenet/'))
#     # for tarinfo in tar_file:
#     #     filename = tarinfo.name
#     #     file_list.append(filename)
#     #     tar_file.extract(tarinfo, os.path.expanduser('~/imagenet_download/'))
