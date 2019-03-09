from configparser import ConfigParser
from datetime import datetime
from os import listdir, mkdir
from sys import stderr, exc_info
from time import time
from torch import device


class GanConfig:
    def __init__(self, config: ConfigParser):
        self._config = config
        config = config['GAN']

        self.train_data = config['TrainData']
        self.test_data = config['TestData']

        self.generator_learning_rate = float(config['GeneratorLearningRate'])
        self.discriminator_learning_rate = float(config['DiscriminatorLearningRate'])

        self.batch_size = int(config['BatchSize'])
        self.scale_factor = int(config['UpscaleFactor'])
        self.iter_limit = int(config['IterationLimit'])

        self.device = device(config['Device'])
        self.identityNN = config['IdentityNN']
        self.shape_predictor = config['ShapePredictor']

        self.generator_module = config['Generator']
        self.discriminator_module = config['Discriminator']
        self.model_name = config['Discriminator']

        assert self.generator_module in listdir('models') \
            and self.discriminator_module in listdir('models'), \
            "Given model name is not in models directory!"

        if "GeneratorSnapshot" in config.keys():
            self.gen_snapshot = config['GeneratorSnapshot']
        else:
            self.gen_snapshot = None

        if "DiscriminatorSnapshot" in config.keys():
            self.disc_snapshot = config['DiscriminatorSnapshot']
        else:
            self.disc_snapshot = None

        self.iter_per_snapshot = int(config['IterationsPerSnapshot'])
        self.iter_per_image = int(config['IterationsPerImage'])
        self.iter_per_eval = int(config['IterationsPerEvaluation'])
        self.test_iter = int(config['TestIterations'])
        # setattr(self, 'IterationsPerImage', int(config['IterationsPerImage']))


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
