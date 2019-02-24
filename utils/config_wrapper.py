from configparser import ConfigParser
from datetime import datetime
from os import listdir, mkdir
from sys import stderr, exc_info
from time import time
from torch import device

class ConfigWrapper:
    def __init__(self, config_path):
        try:
            config = ConfigParser()
            config.read(config_path)
        except Exception:
            print('Can\'t read config file!', file=stderr)
            ex_type, ex_inst, ex_tb = exc_info()
            raise ex_type.with_traceback(ex_inst, ex_tb)

        self._config = config

        config = config['Config']

        self.train_data = config['TrainData']
        self.test_data = config['TestData']

        self.learning_rate = float(config['LearningRate'])
        self.batch_size = int(config['BatchSize'])
        self.scale_factor = int(config['UpscaleFactor'])
        self.iter_limit = int(config['IterationLimit'])

        self.device = device(config['Device'])
        self.identityNN = config['IdentityNN']
        self.shape_predictor = config['ShapePredictor']

        if "snapshot" in config.keys():
            self.snapshot = config['Snapshot']
        else:
            self.snapshot = None

        self.iter_per_snapshot = int(config['IterationsPerSnapshot'])
        self.iter_per_image = int(config['IterationsPerImage'])
        self.iter_per_eval = int(config['IterationsPerEvaluation'])
        self.test_iter = int(config['TestIterations'])
        #setattr(self, 'IterationsPerImage', int(config['IterationsPerImage']))

        self.model_name = config['ModelName']
        assert self.model_name in listdir('models'), "Given model name is not in models directory!"

        # create output directory name
        timestamp = str(datetime.fromtimestamp(time()).strftime('%Y.%m.%d-%H:%M:%S'))
        self.output_folder = config['OutputFolder'] + "/" + self.model_name + "-" + timestamp

        # create directory for results
        try:
            mkdir(self.output_folder)
        except Exception:
            print("Can't create output folder!", file=stderr)
            ex_type, ex_inst, ex_tb = exc_info()
            raise ex_type.with_traceback(ex_inst, ex_tb)

    def save(self, directory):
        with open(directory + "/config.txt", 'w') as file:
            self._config.write(file)
