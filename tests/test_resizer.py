__author__ = 'headradio'
from dldata.metrics.utils import compute_metric
from imagenet.dldatasets import HvM_Categories

import cProfile
dataset = HvM_Categories()
pixel_features = dataset.get_pixel_features()
eval_config = {'train_q': {'synset': dataset.synset_list[:2]},
               'test_q': {'synset': dataset.synset_list[:2]},
               'npc_train': 150, 'npc_test': 50, 'num_splits': 1, 'npc_validate': 0,
               'split_by': 'synset',
               'labelfunc': 'synset',
               'metric_screen': 'classifier',
               'metric_kwargs': {'model_type': 'MCC2', 'normalization': False}}

cProfile.run('print compute_metric(pixel_features, dataset, eval_config)')
