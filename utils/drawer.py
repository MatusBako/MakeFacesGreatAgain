from cv2 import imwrite
import numpy as np


class Drawer:
    def __init__(self, output_folder, scale_factor):
        self.train_cumloss = []
        self.test_cumloss = []
        self.psnr_list = []

        self.scale_factor = scale_factor
        self.output_folder = output_folder

    def save_images(self, data: np.array, result: np.array, target: np.array, label: str):
        img_count = result.data.shape[0]

        limit = 16 if img_count >= 16 else img_count

        height, width, _ = result[0].shape

        stacks = []
        for i in range(0, 16, 4):
            to_stack = []
            for j in range(i, i + 4):
                to_stack.append(data[j] * 256)
                to_stack.append(result[j] * 256)
                to_stack.append(target[j] * 256)

            stacks.append(np.hstack(to_stack))
        collage = np.vstack(stacks)

        imwrite(self.output_folder + "/" + label + ".png", collage[:, :, ::-1])

