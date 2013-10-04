__author__ = 'headradio'
import imagenet.dataset
import random


def test_names():
    """this test is too brittle.  it is currently broken and probably should be eliminated
       and replaced with more informative tests. 
    """
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
        'PixelHardSynsets20_fb902a720218ad6cf75934c09f48f1678cc9c823',
        'PixelHardSynsets_d052b5fa9f7026955c799ce238b096ac00615298',
        'HvM_Categories_e5e3929ca9d206fa0f82767a0bf63c40bf7c586a',
        'Challenge_Synsets_100_Random_5048eda891e0a7ec15381684e593eecfa9dc9234',
        'Challenge_Synsets_2_Pixel_Hard_389367a954e9440e2ecf04aedd32684395600fa4',
        'Imagenet_6eef6648406c333a4035cd5e60d0bf2ecf2606d7']

    assertion = [x == y for x, y in zip(names, true_names)]
    assert all(assertion), ("Datasets don't have correct names:", names, true_names)


def test_meta_PixelHardSynsets20():
    """Is this test correct?  Anyway, we need more like it. 
    """
    dataset = inet.dldatasets.PixelHardSynsets20()
    assert dataset.synset_list = ['n02262449',
                                  'n01773549',
                                  'n03376159',
                                  'n01623425',
                                  'n02152881',
                                  'n01317813',
                                  'n02084071',
                                  'n02356381',
                                  'n03016737',
                                  'n02075296',
                                  'n01772664',
                                  'n03745146',
                                  'n02437616',
                                  'n01887474',
                                  'n02093428',
                                  'n02324587',
                                  'n02441942',
                                  'n02358890',
                                  'n01563746',
                                  'n01540090']


def test_image_source():
    """DY: this test is currently failing for me.   
    """  
    dataset = imagenet.dataset.FullImagenet()
    filenames = dataset.meta['filename']
    source = imagenet.dataset.get_img_source()
    random.seed(0)
    hashes = [imagenet.dataset.get_id(source.get(filename).read())
              for filename in random.sample(filenames, 10)]
    true_hashes = [
        'd20373634e5ab1cd713dd21f646b954d2d062a9c',
        '12fd22d1b7ce2501e99596656cc45802ac3b10e2',
        'a5c789af78b6499bcf96214bc92a629b3b9c014e',
        'a5b3bc12b9bc19791e987f011a1674ba195c7982',
        'b528db7dbc76f8c6a90a6a787b452a8823c73877',
        'dd5af697682b69e2238b1c1a31e7c02a2b37e12c',
        '1c023eebdfd572aa1724ae7773c942209c541635',
        '6339022f04417f2e7590f738e2da6dcdcbf1ccb0',
        '3014ada6f2e5f961e496d778a5078d1ba5d02e83',
        '421b9f33622b47cc37e0653a1f638a2a2cb23f6e']

    assert all([x == y for x, y in zip(hashes, true_hashes)]), 'A small random sample of all files is correct'
