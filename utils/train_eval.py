import numpy as np
from typing import Union

class TrainingEvaluator:
    """ Class for resolving if the net is still training. """

    def __init__(self, size: int):
        assert size % 10 == 0, "Size of evaluator buffer must be divisible by ten!"

        self._size = size

        self._loss = np.zeros(self._size, dtype=np.float)
        self._count = 0

    def append(self, value: Union[float, np.float]):
        # loss array full
        if self._count == self._size:
            self._loss[:-1] = self._loss[1:]
            self._loss[-1] = value
        else:
            self._loss[self._count] = value
            self._count += 1

    def reset(self):
        """ Clear half of the loss array and continue appending after the other half.

        Function is called when learning rate is lowered so that we reset
        storing process and delete half of stored values. This ensures that
        newer values are stored (half the size) before another testing.

        """
        self._count = self._size // 2
        self._loss[:self._size//2] = self._loss[self._size//2:]

    def is_training(self) -> bool:
        """ Function which states if network is still training.

        Returns:
            bool: True if network is still training.
        """
        if self._count < self._size:
            return True

        if np.average(self._loss[self._size//2:]) / np.average(self._loss[:self._size//2]) > 0.99:
            return False

        return True
