from configparser import SectionProxy
from json import loads
from torch import optim
from torch.optim import lr_scheduler

from feature_extractor import DlibFeatureExtractor, senet50_ft


def build_optimizer(cfg_section: SectionProxy, net_params, learning_rate):
    name = cfg_section['Name']
    args = [net_params] + loads(cfg_section['Args']) if cfg_section.get('Args') is not None else []
    kwargs = loads(cfg_section['Kwargs']) if cfg_section.get('Kwargs') is not None else {}
    kwargs['lr'] = learning_rate

    # dynamically get optimizer constructor
    optim_constr = getattr(optim, name)
    return optim_constr(*args, **kwargs)


def build_scheduler(cfg_section: SectionProxy, optimizer):
    name = cfg_section['Name']
    args = [optimizer] + loads(cfg_section['Args']) if cfg_section.get('Args') is not None else []
    kwargs = loads(cfg_section['Kwargs']) if cfg_section.get('Kwargs') is not None else {}

    optim_constr = getattr(lr_scheduler, name)
    return optim_constr(*args, **kwargs)


def build_feature_extractor(cfg_section: SectionProxy):
    name = cfg_section.get('Type')

    if name == "dlib":
        return DlibFeatureExtractor(cfg_section.get('ShapePredictor'), cfg_section.get('Extractor'))
    elif name == "senet":
        network = senet50_ft(weights_path=cfg_section.get('Snapshot'))
        network.eval()
        return network
    else:
        raise Exception("Wrong feature extractor type.")
