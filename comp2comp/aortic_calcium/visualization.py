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

        np.save(
            os.path.join(self.output_dir_images_organs, "calcium_mask.npy"),
            inference_pipeline.calc_mask,
        )
        np.save(
            os.path.join(self.output_dir_images_organs, "ct_scan.npy"),
            inference_pipeline.medical_volume.get_fdata(),
        )

        return {}


class AorticCalciumPrinter(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):

        metrics = inference_pipeline.metrics

        inference_pipeline.csv_output_dir = os.path.join(inference_pipeline.output_dir, "metrics")
        os.makedirs(inference_pipeline.csv_output_dir, exist_ok=True)

        with open(
            os.path.join(inference_pipeline.csv_output_dir, "aortic_calcification.csv"), "w"
        ) as f:
            f.write("Volume (cm^3),Mean HU,Median HU,Max HU\n")
            for vol, mean, median, max in zip(
                metrics["volume"], metrics["mean_hu"], metrics["median_hu"], metrics["max_hu"]
            ):
                f.write("{},{:.1f},{:.1f},{:.1f}\n".format(vol, mean, median, max))

        with open(
            os.path.join(inference_pipeline.csv_output_dir, "aortic_calcification_total.csv"), "w"
        ) as f:
            f.write("Total number,{}\n".format(metrics["num_calc"]))
            f.write("Total volume (cm^3),{}\n".format(metrics["volume_total"]))

        distance = 25
        print("\n")
        if metrics["num_calc"] == 0:
            print("No aortic calcifications were found.")
        else:
            print("Statistics on aortic calcificaitons:")
            print("{:<{}}{}".format("Total number:", distance, metrics["num_calc"]))
            print("{:<{}}{:.3f}".format("Total volume (cm³):", distance, metrics["volume_total"]))
            print(
                "{:<{}}{:.1f}+/-{:.1f}".format(
                    "Mean HU:", distance, np.mean(metrics["mean_hu"]), np.std(metrics["mean_hu"])
                )
            )
            print(
                "{:<{}}{:.1f}+/-{:.1f}".format(
                    "Median HU:",
                    distance,
                    np.mean(metrics["median_hu"]),
                    np.std(metrics["median_hu"]),
                )
            )
            print(
                "{:<{}}{:.1f}+/-{:.1f}".format(
                    "Max HU:", distance, np.mean(metrics["max_hu"]), np.std(metrics["max_hu"])
                )
            )
            print(
                "{:<{}}{:.3f}+/-{:.1f}".format(
                    "Mean volume (cm³):",
                    distance,
                    np.mean(metrics["volume"]),
                    np.std(metrics["volume"]),
                )
            )
            print(
                "{:<{}}{:.3f}".format(
                    "Median volume (cm³):", distance, np.median(metrics["volume"])
                )
            )
            print("{:<{}}{:.3f}".format("Max volume (cm³):", distance, np.max(metrics["volume"])))
            print("{:<{}}{:.3f}".format("Min volume (cm³):", distance, np.min(metrics["volume"])))

        print("\n")

        return {}
