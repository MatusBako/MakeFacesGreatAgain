#!/usr/bin/env python3

from datasets import DatasetFFHQ # DatasetImagenet, DatasetCelebA, DatasetFFHQ

import argparse
from configparser import ConfigParser
from torch.utils.data import DataLoader
from sys import exc_info


def get_config() -> ConfigParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file of net.')
    args = parser.parse_args()

    config = ConfigParser()
    config.optionxform = str
    config.read(args.config)
    return config


def main():
    config = get_config()
    nn_config = config['CNN'] if 'CNN' in config else config['GAN']
    data_config = config['Dataset']

    # dynamically import Solver
    if 'GAN' in config:
        Solver = __import__("models." + nn_config['Discriminator'], fromlist=['Solver']).Solver
    elif 'CNN' in config:
        Solver = __import__("models." + nn_config['ModelName'], fromlist=['Solver']).Solver
    solver = Solver(config)

    # TODO: dynamically choose dataset loader based on config
    Dataset = getattr(__import__("datasets", fromlist=[config['Dataset']['Class']]), config['Dataset']['Class'])
    train_set = Dataset(data_config['TrainData'], nn_config.getint('UpscaleFactor'), length=data_config.getint('TrainLength'))
    test_set = Dataset(data_config['TestData'], nn_config.getint('UpscaleFactor'), length=data_config.getint('TestLength'))

    training_data_loader = DataLoader(dataset=train_set, batch_size=nn_config.getint('BatchSize'), shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=nn_config.getint('BatchSize'), shuffle=True)

    solver.build_models()

    if 'CNN' in config and 'Snapshot' in nn_config:
        solver.load_model(nn_config['Snapshot'])
    elif 'GAN' in config:
        if 'GeneratorSnapshot' in nn_config:
            solver.load_generator_model(nn_config.get('GeneratorSnapshot'))
        if 'DiscriminatorSnapshot' in nn_config:
            solver.load_discriminator_model(nn_config.get('DiscriminatorSnapshot'))

    try:
        solver.train(training_data_loader, testing_data_loader)
    except KeyboardInterrupt as e:
        solver.save_model()
        if solver.logger:
            solver.logger.finish()

        raise e
    except Exception:
        if solver.logger:
            solver.logger.finish()

        ex_type, ex_inst, ex_tb = exc_info()
        raise ex_type.with_traceback(ex_inst, ex_tb)
    finally:
        print("Iterations:", solver.iteration)


if __name__ == "__main__":
    main()
