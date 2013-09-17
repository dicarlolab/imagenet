__author__ = 'headradio'

import hashlib

import numpy as np

import imagenet.dataset
import imagenet.dldatasets as dldatasets


def smoke_test():
    dataset = dldatasets.HvM_Categories()
    print dataset.meta[0]


def test_Challenge_Synsets_20_Pixel_Hard_meta():
    dataset = dldatasets.Challenge_Synsets_20_Pixel_Hard()
    meta = dataset.meta
    assert meta.dtype.names == ('filename', 'synset')
    magg = meta.aggregate(On=['synset'], AggFunc=len)
    assert magg.tolist() == [('n02091831', 1087),
                             ('n02093428', 1329),
                             ('n02096177', 1377),
                             ('n02097474', 774),
                             ('n02102318', 1606),
                             ('n02109047', 1575),
                             ('n02236044', 1499),
                             ('n02795169', 1171),
                             ('n03196217', 1097),
                             ('n03259280', 1313),
                             ('n03424325', 1242),
                             ('n03633091', 1817),
                             ('n03666591', 1599),
                             ('n03804744', 1121),
                             ('n03950228', 1871),
                             ('n04153751', 1221),
                             ('n04200800', 1201),
                             ('n04265275', 949),
                             ('n04507155', 1341),
                             ('n04599235', 1640)]
    fn = '|'.join(meta['filename'])
    assert hashlib.sha1(fn).hexdigest() == '2af9b9fe56963cc2d1643f5f7e8d1659a45ec68e'


def test_Challenge_Synsets_20_Pixel_Hard_imgs():
    #diego do you want this test somewhere else?

    dataset = dldatasets.Challenge_Synsets_20_Pixel_Hard()
    imgs = dataset.get_images(preproc={'resize_to': (256, 256),
                                        'mode': 'RGB',
                                        'dtype': 'float32',
                                        'normalize': False,
                                        'crop': None,
                                        'mask': None})

    inds = np.arange(0, imgs.shape[0], 1000)
    hashes = [hashlib.sha1(imgs[i]).hexdigest() for i in inds]
    print hashes
    assert hashes == IMG_HASHES_HARD_20


IMG_HASHES_HARD_20 = ['f4c09295dd8eab54f77afeaf63a173937a5e5a81',
 'a0c675d849e89c46b2d048fccce03d483f064c09',
 '3bb98cab79a21e7860dcefe47cefd285fdc6d4d7',
 '9178a2af2ef6837ac22581fa3fa053dfb235ad0d',
 'ad2eebfe1247fe1acad2bb3648679b89ca2c9a0d',
 '61c3697e6965fd82c314c6b42ec6dfce7d8f6deb',
 '0d0a23dfa72b0e90da67b6c78625db80200881a4',
 'd1a8c863eb3f55f553c66126e516b2d62bf3bed1',
 '20765fd9667270018822e2bd0009f1b502debd4c',
 'ecea7df1b7a4ac6a118d39f7d66862528fbd29a9',
 '2b6e9c772ded1c84a28128d1e5e47039d976cd83',
 'f2a084ece9933972ba6a4b39a926a187b2f82914',
 '01356123f36050ef87e584faca83aae639465e00',
 '91a1168fe77e74edbd422713e7faf54d39951847',
 '7a12da5ae35832d42786df2fc4e28e07798a3c91',
 '70cbdf10f9958adaffaef3ddd9aef546d02028d0',
 '411b675c5a3fd3f3c44be0d6e7bf6eeab7ab5cf0',
 '20c183cc21e4829e250fca31b39aa323736d6b5f',
 '343927ca985f5566ce8b55530aa2760a8a61f9b7',
 '51670f3a3f5dfdcdba1a4251f23660055e673df8',
 '7437a5d8b23039f515770b35f771237e6d55454d',
 '6fb7dd0760d9f1937fe243fa527cde56021f6a53',
 '0a5c86362976ffdde1b2eb62c6143f94a1a3a14f',
 '4992ce125b0dc0b0270c4297b6b9beb1ec29d4cc',
 '26f46720f40b13dc697b4cb4af5a1658dc27f6ec',
 'd7fd27877598c1dff97c37df62bbccbe31cb317e',
 '531d5ecb492d9f5a1dfa8c47ef8c86fd3f0b637f']
