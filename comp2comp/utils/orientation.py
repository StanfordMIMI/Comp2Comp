import nibabel as nib

from comp2comp.inference_class_base import InferenceClass

class ToCanonical(InferenceClass):
    """Convert spine segmentation to canonical orientation."""
    def __init__(self):
        super().__init__()

    def __call__(self, inference_pipeline):
        """
        First dim goes from L to R.
        Second dim goes from P to A.
        Third dim goes from I to S.
        """
        inference_pipeline.flip_si = False  # necessary for finding dicoms in correct order
        if "I" in nib.aff2axcodes(inference_pipeline.medical_volume.affine):
            inference_pipeline.flip_si = True

        canonical_segmentation = nib.as_closest_canonical(inference_pipeline.segmentation)
        canonical_medical_volume = nib.as_closest_canonical(inference_pipeline.medical_volume)

        inference_pipeline.segmentation = canonical_segmentation
        inference_pipeline.medical_volume = canonical_medical_volume
        inference_pipeline.pixel_spacing_list = canonical_medical_volume.header.get_zooms()
        return {}