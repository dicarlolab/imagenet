__author__ = 'headradio'
import random
import time
import imagenet


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
    #Slow to initialize and probably will not be reused anywhere
    #dataset = imagenet.dldatasets.Challenge_Synsets_100_Random()
    #names.append(dataset.specific_name)
    dataset = imagenet.dldatasets.Challenge_Synsets_2_Pixel_Hard()
    names.append(dataset.specific_name)
    dataset = imagenet.dataset.Imagenet()
    names.append(dataset.specific_name)
    true_names = ['PixelHardSynsets20_5187c7eddfa35093111e22705310e385d36f8ecf',
                  'PixelHardSynsets_75372977ac90b6148148f3286d31d26ace3d578e',
                  'HvM_Categories_556f92cd5da08b1ffb20d7ede4c6581b0c440fac',
                  #'Challenge_Synsets_100_Random_5048eda891e0a7ec15381684e593eecfa9dc9234',
                  'Challenge_Synsets_2_Pixel_Hard_389367a954e9440e2ecf04aedd32684395600fa4',
                  'Imagenet_6eef6648406c333a4035cd5e60d0bf2ecf2606d7']

    assertion = [x == y for x, y in zip(names, true_names)]
    assert all(assertion), ("Datasets don't have correct names:", names, true_names)


def test_meta_PixelHardSynsets20():
    """Is this test correct?  Anyway, we need more like it.
    """
    dataset = imagenet.dldatasets.PixelHardSynsets20()
    assert dataset.synset_list == ['n02262449',
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


def test_PixelHardSynsets20_hardness():
    """
    this test should try to actively ensure that the synsets as listed are actually hard.

    To do this, you could pick (e.g.) 10 reference synsets.
    Then using MCC classifier, compute the binary separation for the top few synsets in
    PixelHardSynsets20.synset_list against each of the  10 reference synsets.   Also
    do so for a few randomly chosen synsets NOT in the top 20 hardest.
    Then assert that the classification results for the hard sets are worse than the randomly chosen sets.
    *you probably also want to assert that you get the correct expected accuracy values for each of these
    classification problems*
    """
    pass


def test_image_source():
    """DY: this test is currently failing for me.
    """
    dataset = imagenet.dataset.Imagenet()
    filenames = dataset.meta['filename']
    source = imagenet.dataset.get_img_source()
    random.seed(0)
    hashes = [imagenet.dataset.get_id(source.get(filename).read())
              for filename in random.sample(filenames, 10)]
    true_hashes = ['4e232d0da8aa762bbc6e809efe8a2c62a495082b', '8596ddc0fb79310e6bf7ca6492a9f3b7903fd9b0',
                   'b3ae581f50f588c57d9ec7e1651b134049fa0f56', '143d57ffc9c1cffbf1e0bcc46fe4ba0978c19fa4',
                   'f3d1f193063e8255733fc6f9dd20a52fe5efa454', '3a452495475634337f014807e435af5d06d26511',
                   '8529790f0d6a3b6ebf1bda0f1eccf3dfd3b0f421', '4eb48263082ee816ad234ee46b51e0af008dc9d2',
                   '2300f189ada816dc7b25213427288debe8be9672', 'c4d9b0a51509b08d3fca170e96eea8b1ee259c83']

    assert all([x == y for x, y in zip(hashes, true_hashes)]), (
        'A small random sample of all files is correct', hashes, true_hashes)


def test_get_images():
    dset = imagenet.dldatasets.PixelHardSynsets2013Challenge()
    preproc = {'crop': None,
               'dtype': 'float32',
               'mask': None,
               'mode': 'RGB',
               'normalize': True,
               'resize_to': (64, 64)}
    imgs = dset.get_images(preproc=preproc)
    t = time.time()
    X = imgs[:100]
    t0 = time.time() - t

    imgs1 = dset.get_images(preproc=preproc, cache=True)

    t = time.time()
    X1 = imgs1[:100]
    t1 = time.time() - t

    assert (X == X1).all()
    print('Time to get 100 images, no cache: %f' % t0)
    print('Time to get 100 images, cache -- first time (might already be cached): %f' % t1)
