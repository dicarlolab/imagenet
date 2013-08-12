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
        Imagenet_filename_subset.__init__(self, filenames, name, img_path)


#this is the dataset we used in our initial model screen
class Model_Screen_1(Imagenet_filename_subset):
    def __init__(self, img_path=default_image_path):
        synsets = get2013_Categories()
        num_per_synset = 250
        name = 'Model_Screen_1'
        full_dict = get_full_filename_dictionary()
        filenames = []
        map(filenames.extend, [random.sample(full_dict[synset], num_per_synset) for synset in synsets])
        Imagenet_filename_subset.__init__(self, filenames, name, img_path)
