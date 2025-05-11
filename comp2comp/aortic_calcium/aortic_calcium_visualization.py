import os

import numpy as np

from comp2comp.inference_class_base import InferenceClass
from comp2comp.aortic_calcium.visualization_utils import createMipPlot, createCalciumMosaic, mergeMipAndMosaic

class AorticCalciumVisualizer(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_images_organs = os.path.join(self.output_dir, "images/")
        inference_pipeline.output_dir_images_organs = self.output_dir_images_organs

        if not os.path.exists(self.output_dir_images_organs):
            os.makedirs(self.output_dir_images_organs)
        
        # Create MIP part of the overview plot
        createMipPlot(
            inference_pipeline.ct, 
            inference_pipeline.calc_mask,
            inference_pipeline.aorta_mask,
            inference_pipeline.t12_plane == 1,
            inference_pipeline.calcium_threshold,
            inference_pipeline.pix_dims,
            inference_pipeline.metrics,
            self.output_dir_images_organs,
        )
        
        ab_num = inference_pipeline.metrics['Abdominal']['num_calc'] 
        th_num = inference_pipeline.metrics['Thoracic']['num_calc'] 
        # Create mosaic part of the overview plot
        if not (ab_num == 0 and th_num == 0):      
            createCalciumMosaic(
                inference_pipeline.ct, 
                inference_pipeline.calc_mask,
                inference_pipeline.dilated_aorta_mask, # the dilated mask is used here
                inference_pipeline.spine_mask,
                inference_pipeline.pix_dims,
                self.output_dir_images_organs,
                inference_pipeline.args.mosaic_type,
            )
        
        # Merge the two images created above for the final report 
        mergeMipAndMosaic(
            self.output_dir_images_organs
        )
        
        return {}


class AorticCalciumPrinter(InferenceClass):
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):

        all_metrics = inference_pipeline.metrics

        inference_pipeline.csv_output_dir = os.path.join(
            inference_pipeline.output_dir, "metrics"
        )
        os.makedirs(inference_pipeline.csv_output_dir, exist_ok=True)

        # Write metrics to CSV file
        with open(
            os.path.join(inference_pipeline.csv_output_dir, "aortic_calcification.csv"),
            "w",
        ) as f:
            f.write("Volume (cm^3),Mean HU,Median HU,Max HU\n")

        with open(
            os.path.join(inference_pipeline.csv_output_dir, "aortic_calcification.csv"),
            "a",
        ) as f:

            for region, metrics in all_metrics.items():
                f.write(region + ",,,\n")

                for vol, mean, median, max in zip(
                    metrics["volume"],
                    metrics["mean_hu"],
                    metrics["median_hu"],
                    metrics["max_hu"],
                ):
                    f.write("{},{:.1f},{:.1f},{:.1f}\n".format(vol, mean, median, max))

        # Write total results
        with open(
            os.path.join(
                inference_pipeline.csv_output_dir, "aortic_calcification_total.csv"
            ),
            "w",
        ) as f:
            for region, metrics in all_metrics.items():
                f.write(region + ",\n")

                f.write("Total number,{}\n".format(metrics["num_calc"]))
                f.write("Total volume (cm^3),{:.3f}\n".format(metrics["volume_total"]))
                f.write(
                    "Threshold (HU),{:.1f}\n".format(
                        inference_pipeline.calcium_threshold
                    )
                )

                f.write(
                    "{},{:.1f}+/-{:.1f}\n".format(
                        "Mean HU",
                        np.mean(metrics["mean_hu"]),
                        np.std(metrics["mean_hu"]),
                    )
                )
                f.write(
                    "{},{:.1f}+/-{:.1f}\n".format(
                        "Median HU",
                        np.mean(metrics["median_hu"]),
                        np.std(metrics["median_hu"]),
                    )
                )
                f.write(
                    "{},{:.1f}+/-{:.1f}\n".format(
                        "Max HU",
                        np.mean(metrics["max_hu"]),
                        np.std(metrics["max_hu"]),
                    )
                )
                f.write(
                    "{},{:.3f}+/-{:.3f}\n".format(
                        "Mean volume (cm³):",
                        np.mean(metrics["volume"]),
                        np.std(metrics["volume"]),
                    )
                )
                f.write(
                    "{},{:.3f}\n".format(
                        "Median volume (cm³)", np.median(metrics["volume"])
                    )
                )
                f.write(
                    "{},{:.3f}\n".format("Max volume (cm³)", np.max(metrics["volume"]))
                )
                f.write(
                    "{},{:.3f}\n".format("Min volume (cm³):", np.min(metrics["volume"]))
                )
                f.write(
                    "{},{:.3f}\n".format("% Calcified aorta:", metrics["perc_calcified"])
                )

                if inference_pipeline.args.threshold == "agatston":
                    f.write("Agatston score,{:.1f}\n".format(metrics["agatston_score"]))

        distance = 25
        print("\n")
        print("Statistics on aortic calcifications:")

        for region, metrics in all_metrics.items():
            print(region + ":")

            if metrics["num_calc"] == 0:
                print("No aortic calcifications were found.\n")
            else:
                print("{:<{}}{}".format("Total number:", distance, metrics["num_calc"]))
                print(
                    "{:<{}}{:.3f}".format(
                        "Total volume (cm³):", distance, metrics["volume_total"]
                    )
                )
                print(
                    "{:<{}}{:.1f}+/-{:.1f}".format(
                        "Mean HU:",
                        distance,
                        np.mean(metrics["mean_hu"]),
                        np.std(metrics["mean_hu"]),
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
                        "Max HU:",
                        distance,
                        np.mean(metrics["max_hu"]),
                        np.std(metrics["max_hu"]),
                    )
                )
                print(
                    "{:<{}}{:.3f}+/-{:.3f}".format(
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
                print(
                    "{:<{}}{:.3f}".format(
                        "Max volume (cm³):", distance, np.max(metrics["volume"])
                    )
                )
                print(
                    "{:<{}}{:.3f}".format(
                        "Min volume (cm³):", distance, np.min(metrics["volume"])
                    )
                )
                print(
                    "{:<{}}{:.3f}".format(
                        "Threshold (HU):",
                        distance,
                        inference_pipeline.calcium_threshold,
                    )
                )
                print(
                    "{:<{}}{:.3f}".format(
                        "% Calcified aorta",
                        distance,
                        metrics["perc_calcified"],
                    )
                )
                
                if inference_pipeline.args.threshold == "agatston":
                    print(
                        "{:<{}}{:.1f}".format(
                            "Agatston score:", distance, metrics["agatston_score"]
                        )
                    )

                print("\n")

        return {}
