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
                f.write(region + ',,,\n')
                
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
                f.write(region + ',\n')
    
                f.write("Total number,{}\n".format(metrics["num_calc"]))
                f.write("Total volume (cm^3),{}\n".format(metrics["volume_total"]))
                f.write("Threshold (HU),{}\n".format(inference_pipeline.calcium_threshold))

        distance = 25
        print("\n")
        print("Statistics on aortic calcifications:")
        
        for region, metrics in all_metrics.items():
            print(region + ':')
    
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
                        "Threshold (HU):", distance, inference_pipeline.calcium_threshold
                    )
                )
    
                print("\n")

        return {}
