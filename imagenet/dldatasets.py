import itertools
import random
import numpy as np
from dataset import (Imagenet_filename_subset, 
                     Imagenet_synset_subset,
                     get2013_Categories,
                     Imagenet,
                     broken_synsets,
                     default_fs)


class HvM_Categories(Imagenet_filename_subset):
    """
    Hand-chosen imagenet equivalents of HvM categories
    Has an attribute called translation dict explaining the mapping
    """
    def __init__(self):
        full_dict = self.get_full_filenames_dictionary()
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
        synsets = random.sample(list(set(get2013_Categories())-set(broken_synsets)), 100)
        num_per_synset = 200
        full_dict = self.get_full_filenames_dictionary()
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

    def get_hmo_feats0(self):
        grid_file = default_fs.get('Challenge_synsets_20_Pixel_hard_800.npy')
        feats = np.load(grid_file)
        return feats


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
        screen_synsets = list((set(get2013_Categories()) | set(big_synsets)) - broken_synsets)
        full_filenames_dict = self.get_full_filenames_dictionary()
        filenames = []
        filenames_dict = {}
        num_per_synset = 200
        for synset in screen_synsets:
            filenames_from_synset = random.sample(full_filenames_dict[synset], num_per_synset)
            filenames.extend(filenames_from_synset)
            filenames_dict[synset] = filenames_from_synset
        data = {'filenames': filenames,
                'filenames_dict': filenames_dict}
        super(Big_Pixel_Screen, self).__init__(data=data)
