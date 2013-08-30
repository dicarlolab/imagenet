from dataset import *


class HvM_Categories(Imagenet_filename_subset):
    """
    Hand-chosen imagenet equivalents of HvM categories
    Has an attribute called translation dict explaining the mapping
    """
    def __init__(self, img_path=default_image_path):
        name = 'HvM_Categories_Approximated_by_Synsets'
        full_dict = get_full_filename_dictionary()
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
        super(HvM_Categories, self).__init__(filenames, name, img_path)

    def memmap_pixel_features(self):
        import os.path as path
        filename = path.join(self.img_path, 'HvM.dat')
        return np.memmap(filename, 'float32', 'w+')


#this is the dataset we used in pixel screening
class Challenge_Synsets_100_Random(Imagenet_filename_subset):
    def __init__(self, img_path=default_image_path, meta_path=None):
        name = 'challenge_synsets_100_random'
        if meta_path is None:
            meta_path = os.path.join(default_meta_path, name)
        self.meta_path = meta_path
        random.seed(name)
        synsets = random.sample(get2013_Categories(), 100)
        num_per_synset = 200
        full_dict = get_full_filename_dictionary()
        filenames = []
        for synset in synsets:
            filenames.extend(random.sample(full_dict[synset], num_per_synset))
        super(Challenge_Synsets_100_Random, self).__init__(filenames, name, img_path, meta_path)



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
