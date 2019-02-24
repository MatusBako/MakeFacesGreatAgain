#!/usr/bin/env python3

from datasets import DatasetImagenet, DatasetCelebA
from utils import ConfigWrapper

import argparse
from torch.utils.data import DataLoader
from sys import exc_info


def get_config() -> ConfigWrapper:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file of net.')
    args = parser.parse_args()

    return ConfigWrapper(args.config)


def main():
    config = get_config()

    # dynamically import Solver
    Solver = __import__("models." + config.model_name, fromlist=['Solver']).Solver
    solver = Solver(config)

    train_set = DatasetCelebA(config.train_data, config.scale_factor)
    test_set = DatasetCelebA(config.test_data, config.scale_factor)
    #train_set = DatasetImagenet(config.train_data, config.scale_factor)
    #test_set = DatasetImagenet(config.test_data, config.scale_factor)

    training_data_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=config.batch_size, shuffle=True)

    solver.build_models(config)

    # TODO: model loading
    if config.snapshot is not None:
        solver.load_model(config.snapshot)

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
