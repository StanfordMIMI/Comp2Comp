
import os

import numpy as np

from comp2comp.inference_class_base import InferenceClass

class AorticCalciumVisualizer(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):

        self.output_dir = inference_pipeline.output_dir
        self.output_dir_images_organs = os.path.join(self.output_dir, "images/")
        inference_pipeline.output_dir_images_organs = self.output_dir_images_organs

        if not os.path.exists(self.output_dir_images_organs):
            os.makedirs(self.output_dir_images_organs)

        np.save(os.path.join(self.output_dir_images_organs, 'calcium_mask.npy'), inference_pipeline.calc_mask)
        np.save(os.path.join(self.output_dir_images_organs, 'ct_scan.npy'), inference_pipeline.medical_volume.get_fdata())

        return {}
