from configparser import SectionProxy
from json import loads
from torch import optim
from torch.optim import lr_scheduler

from feature_extractor import DlibFeatureExtractor, Senet50FeatureExtractor, VggFeatureExtractor


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


# TODO: remove, create FE dynamically based on config (=> one layer removed)
def build_feature_extractor(cfg_section: SectionProxy):
    name = cfg_section.get('Type').lower()

    if name == "dlib":
        return DlibFeatureExtractor(cfg_section.get('ShapePredictor'), cfg_section.get('Extractor'))
    elif name == "senet":
        return Senet50FeatureExtractor(cfg_section.get("Detections"), cfg_section.get('Snapshot'))
    elif name == "vgg":
        return VggFeatureExtractor(cfg_section.getint("MaxpoolIdx"), cfg_section.getint("NoActivation"))
    else:
        raise Exception("Wrong feature extractor type.")
