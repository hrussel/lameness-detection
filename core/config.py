import yaml
import argparse

"""
Author: Helena Russello (helena@russello.dev)
"""

class MyConfig(object):

    def __init__(self, args):
        """
         A custom class for reading and parsing a YAML configuration file.

        :param config_path: the path of the configuration file
        """
        config_path = args.config

        self.config = None
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)
            self.gait_scores_csv = self.config['gait_scores_csv']
            self.keypoints_path = self.config['keypoints_path']
            self.data_path = self.config['data_path']
            self.joints = self.config['joints']
            self.features = self.config['features']
            self.merging = self.config['merging']
            self.load_model = self.config['load_model']
            self.save_path = self.config['save_path']
            self.model_type = self.config['model_type']
            self.n_folds = self.config['n_folds']
            self.use_kp = self.config['use_kp']
            self.smoothing = self.config['smoothing']
            self.smoothing_params = self.config['smoothing_params']
            self.flat_cv = self.config['flat_cv']

        if args.model_type:
            self.model_type = args.model_type
        if args.load_model:
            self.load_model = args.load_model
        if args.merging:
            self.merging = args.merging
        if args.features:
            print(args.features)
            self.features = args.features
        if args.flat_cv:
            self.flat_cv = args.flat_cv

    def __str__(self):
        return str(self.config)


def parse_args(description):
    """
    Parse arguments and process the configuration file
    :return: the config and the arguments
    """
    parser = argparse.ArgumentParser(description=description)
    # config file
    parser.add_argument('--config',
                        help='YAML configuration file',
                        default="cfg/config.yml",
                        type=str)
    parser.add_argument('--model_type',
                        help='Classifier type',
                        type=str)
    parser.add_argument('--load_model',
                        help='Model to load',
                        type=str)
    parser.add_argument('--merging',
                        help='How to merge the scores',
                        type=str)
    parser.add_argument('--features',
                        help='List of the features to include',
                        nargs='*',
                        default=None)
    parser.add_argument('--flat_cv',
                        help='Whether to perform flat or nested cv',
                        type=str)

    args, rest = parser.parse_known_args()
    print(args)

    cfg = MyConfig(args)
    print(cfg)

    args = parser.parse_args()  # parse the rest of the args

    return cfg, args
