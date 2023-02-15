import os

class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        if dataset == 'cweeds':
            return str(ROOT_DIR) + '/Datasets/Carrotweed'  # folder that contains VOCdevkit/.
        elif dataset == 'bweeds':
            return str(ROOT_DIR) + '/Datasets/Bonirob_sugarbeetweed'
        elif dataset == 'rweeds':
            return str(ROOT_DIR) + '/Datasets/Rice_seedlingWeed'
        elif dataset == 'sweeds':
            return str(ROOT_DIR) + '/Datasets/sunflower_weed'  # folder that contains dataset/.
        elif dataset == 'svweeds':
            return str(ROOT_DIR) + '/Datasets/sun_bg_light'  # folder that contains dataset/.
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
