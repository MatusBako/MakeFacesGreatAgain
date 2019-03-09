from configparser import ConfigParser
from .cnn_config import CnnConfig
from .gan_config import GanConfig
from sys import stderr, exc_info


def parse_config(config_path: str):
    try:
        config = ConfigParser()
        config.read(config_path)
    except Exception:
        print('Can\'t read config file!', file=stderr)
        ex_type, ex_inst, ex_tb = exc_info()
        raise ex_type.with_traceback(ex_inst, ex_tb)

    if 'CNN' in config.keys():
        return CnnConfig(config)
    elif 'GAN' in config.keys():
        return GanConfig(config)
    else:
        print('Wrong root tag in config file!', file=stderr)
        raise Exception('Wrong root tag in config file!')
