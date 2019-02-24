'''
from os import listdir
from os.path import dirname, basename, isfile
import glob

folder = dirname(__file__)
modules = list(filter(lambda m: not m.endswith(('.py','__pycache__')), listdir(folder)))

__all__ = [ (m[:-3] if m.endswith('.py') else m) for m in modules]

from .abstract_cnn_solver.py import AbstractCnnSolver
'''