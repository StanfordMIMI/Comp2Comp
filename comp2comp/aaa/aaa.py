import os
import zipfile
from pathlib import Path
from time import time
from typing import Union
import matplotlib.pyplot as plt

import dosma
import numpy as np
import wget
import cv2
import scipy.misc
from PIL import Image

import dicom2nifti
import math

from totalsegmentator.libs import (
    download_pretrained_weights,
    nostdout,
    setup_nnunet,
)

from comp2comp.inference_class_base import InferenceClass
from comp2comp.models.models import Models
from comp2comp.spine import spine_utils
import nibabel as nib

class LoadAortaSegmentation(InferenceClass):
    def __init__(self):
        super().__init__()
        # self.dicom_dir = Path("/Users/adritrao/Downloads/1.2.840.4267.32.262103052288294460316908563757299651875/1.2.840.4267.32.151706896185626640902070342803170532046")
        # self.dr = dosma.DicomReader()



    def __call__(self, inference_pipeline):

        dicom2nifti.dicom_series_to_nifti('/Users/adritrao/Downloads/1.2.840.4267.32.262103052288294460316908563757299651875/1.2.840.4267.32.151706896185626640902070342803170532046', '/Users/adritrao/Downloads/1.2.840.4267.32.262103052288294460316908563757299651875/1.2.840.4267.32.151706896185626640902070342803170532046.nii.gz')

        # medical_volume = self.dr.load(self.dicom_dir, group_by=None, sort_by="InstanceNumber")[0]


        seg = nib.load("/Users/adritrao/Downloads/Comp2Comp/outputs/aorta.nii.gz")
        seg = nib.as_closest_canonical(seg)
        seg = seg.get_fdata()

        medical_volume = nib.load("/Users/adritrao/Downloads/1.2.840.4267.32.262103052288294460316908563757299651875/1.2.840.4267.32.151706896185626640902070342803170532046.nii.gz")
        medical_volume = nib.as_closest_canonical(medical_volume)
        medical_volume = medical_volume.get_fdata()

        # seg = nib.as_closest_canonical(seg)
        # medical_volume = nib.as_closest_canonical(medical_volume)
        
        # seg = np.transpose(seg, [1, 0, 2])
        # seg = np.flip(seg, [1, 2])
        
        # aorta_model = Models.model_from_name("ts_aorta")

        # seg = (seg == 7).astype(int)
        print(seg)

        # aorta_model = Models.model_from_name("ts_aorta")
        # seg = (seg == aorta_model.categories["aorta"]).astype(int)

        # print(self.aorta_model["aorta_label"])

        
        axial_masks = []
        ct_image = []

        for i in range(seg.shape[2]):
            axial_masks.append(seg[:, :, i])
        
        for i in range(medical_volume.shape[2]):
            ct_image.append(medical_volume[:, :, i])


        # another thing you could do is save to inference_pipeline
        inference_pipeline.axial_masks = axial_masks

        # save a numpy array for the input CT image
        inference_pipeline.ct_image = ct_image


        return {}



class AortaSegmentation(InferenceClass):
    """Spine segmentation."""

    def __init__(self, input_path):
        super().__init__()
        self.input_path = input_path
        self.model_name = "ts_aorta"
        self.aorta_model = Models.model_from_name(self.model_name)

    def __call__(self, inference_pipeline):
        inference_pipeline.dicom_series_path = self.input_path
        self.output_dir = inference_pipeline.output_dir
        self.output_dir_segmentations = os.path.join(self.output_dir, "segmentations/")
        if not os.path.exists(self.output_dir_segmentations):
            os.makedirs(self.output_dir_segmentations)

        self.model_dir = inference_pipeline.model_dir

        seg, mv = self.aorta_seg(
            self.input_path,
            self.output_dir_segmentations + "aorta.nii.gz",
            inference_pipeline.
            model_dir,
        )

       
        seg = seg.get_fdata()


        medical_volume = mv.get_fdata()

        # fix me
        # seg = (seg == self.aorta_model.categories["aorta"]).astype(int)
        seg = np.where(seg == 7, 1, 0)
        # if self.aorta_model["aorta_label"] hold the value correponsing to the aorta label, then you do the above

        # first dimension is left to right, second dimension is anterior to posterior, third dimension is inferior to superior
        # if the above is correct
    
        axial_masks = []
        ct_image = []

        for i in range(seg.shape[2]):
            axial_masks.append(seg[:, :, i])
        
        for i in range(medical_volume.shape[2]):
            ct_image.append(medical_volume[:, :, i])


        # another thing you could do is save to inference_pipeline
        inference_pipeline.axial_masks = axial_masks

        # save a numpy array for the input CT image
        inference_pipeline.ct_image = ct_image

        # or you can return like this
        # return {"axial_masks": axial_masks}
        return {}


    def aorta_seg(self, input_path: Union[str, Path], output_path: Union[str, Path], model_dir):
        """Run spine segmentation.

        Args:
            input_path (Union[str, Path]): Input path.
            output_path (Union[str, Path]): Output path.
        """

        print("Segmenting spine...")
        st = time()
        os.environ["SCRATCH"] = self.model_dir

        # Setup nnunet
        model = "3d_fullres"
        folds = [0]
        trainer = "nnUNetTrainerV2_ep4000_nomirror"
        crop_path = None
        task_id = [251]

        if self.model_name == "ts_aorta":
            setup_nnunet()
            download_pretrained_weights(task_id[0])
        elif self.model_name == "stanford_spine_v0.0.1":
            self.setup_nnunet_c2c(model_dir)
            self.download_spine_model(model_dir)
        else:
            raise ValueError("Invalid model name.")

        from totalsegmentator.nnunet import nnUNet_predict_image

        with nostdout():

            img, seg = nnUNet_predict_image(
                input_path,
                output_path,
                task_id,
                model=model,
                folds=folds,
                trainer=trainer,
                tta=False,
                multilabel_image=True,
                resample=1.5,
                crop=None,
                crop_path=crop_path,
                task_name="total",
                nora_tag=None,
                preview=False,
                nr_threads_resampling=1,
                nr_threads_saving=6,
                quiet=False,
                verbose=False,
                test=0,
            )
        end = time()

        # Log total time for spine segmentation
        print(f"Total time for spine segmentation: {end-st:.2f}s.")

        return seg, img


class AortaDiameter(InferenceClass):
    # def __init__():
    #     super.__init__()

    def __call__(self, inference_pipeline):
        # get axial_masks from inference_pipeline
        # do stuff
        axial_masks = inference_pipeline.axial_masks

        # add your code

        #either save things to the inference_pipeline or return them for the next class
        return {}

class AortaVisualizer(InferenceClass):

    def normalize_img(self, img: np.ndarray) -> np.ndarray:
        """Normalize the image.
        Args:
            img (np.ndarray): Input image.
        Returns:
            np.ndarray: Normalized image.
        """
        return (img - img.min()) / (img.max() - img.min())
    # def __init__():
    #     super.__init__()

    def __call__(self, inference_pipeline):
        # get axial_masks from inference_pipeline
        # do stuff

        # SLICE_NUM = 380
        # SEG_NUM = 200


        axial_masks = inference_pipeline.axial_masks # list of 2D numpy arrays of shape (512, 512)
        ct_img = inference_pipeline.ct_image # 3D numpy array of shape (512, 512, num_axial_slices)
        # add your code

        # saggital_masks = np.rot90(inference_pipeline.axial_masks)
        # saggital_img = np.rot90(inference_pipeline.ct_image)

        
        

        # image output directory 
        # output_dir = inference_pipeline.output_dir
        # output_dir_images = os.path.join(output_dir, "images/")
        # if not os.path.exists(output_dir_images):
        #     os.makedirs(output_dir_images)

        # plt.imshow(ct_img[SLICE_NUM], cmap="gray")
        # plt.savefig(output_dir_images + "ct_img.png")
        # plt.imshow(axial_masks[SLICE_NUM], alpha=0.5)
        # plt.savefig(output_dir_images + "axial_masks.png")

        # img = ct_img[:, :, 130]
        # img = np.array(ct_img[:, :, 130].astype('uint8'), copy=False) / 255.0
        # img = scipy.misc.toimage(ct_img[:, :, 130])

        SLICE_COUNT = len(ct_img)
        
        for i in range(len(ct_img)):

            mask = axial_masks[i].astype('uint8')
            # img = cv2.imread(output_dir_images + "ct_img.png")

            img = ct_img[i]

            img = np.clip(img, -300, 1800)
            img = self.normalize_img(img) * 255.0
            img = img.reshape((img.shape[0], img.shape[1], 1))
            img = np.tile(img, (1, 1, 3))




            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            # print("Contour: "+str(contour))

    

            if len(contours) != 0:
                # for i in range(len(contours)):
                #   if len(contours[i]) >= 5:

                    areas = [cv2.contourArea(c) for c in contours]
                    sorted_areas = np.sort(areas)

                    contours = contours[areas.index(sorted_areas[-1])]

                    overlay = img.copy()

                    back = img.copy()
                    cv2.drawContours(back, [contours], 0, (0,0,255), -1)

                    # blend with original image
                    alpha = 0.25
                    img = cv2.addWeighted(img, 1-alpha, back, alpha, 0)


                    # cv2.drawContours(img, [contours], -1, (0,255,0), 1)
                    # cv2.fillPoly(img, pts =[contours], color=(158, 157, 212))
                    # res = cv2.addWeighted(img, 0.5, poly, 0.5, 1.0)

                    # alpha = 0.2

                    # img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)



                    
                    ### ELLIPSE
                    ellipse = cv2.fitEllipse(contours)
                    (xc,yc),(d1,d2),angle = ellipse
            
                    cv2.ellipse(img, ellipse, (0, 255, 0), 1)
                    

                    xc, yc = ellipse[0]
                    cv2.circle(img, (int(xc),int(yc)), 5, (0, 0, 255), -1)

                    rmajor = min(d1,d2)/2
                    # rmajor = max(d1,d2)/2
                    if angle > 90:
                        angle = angle - 90
                    else:
                        angle = angle + 90
                    print(angle)
                    xtop = xc + math.sin(math.radians(angle))*rmajor
                    ytop = yc + math.cos(math.radians(angle))*rmajor
                    xbot = xc + math.sin(math.radians(angle+180))*rmajor
                    ybot = yc + math.cos(math.radians(angle+180))*rmajor
                    line = cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 0, 255), 2)

                    pixel_length = math.sqrt( (xtop-xbot)**2 + (ytop-ybot)**2 )

                    area_px = cv2.contourArea(contours)
                    area_mm = area_px
                    # area_mm = round(area_px*RATIO_PIXEL_TO_MM)
                    area_cm = area_px
                    
                    # diameter_mm = round((pixel_length)*RATIO_PIXEL_TO_MM)
                    diameter_mm = pixel_length
                    diameter_cm = pixel_length

                    h,w,c = img.shape
                    lbls = ["Area (mm): "+str(area_mm)+"mm", "Area (cm): "+str(area_cm)+"cm", "Diameter (mm): "+str(diameter_mm)+"mm", "Diameter (cm): "+str(diameter_cm)+"cm", "Slice: "+str(SLICE_COUNT-(i))]
                    offset = 0
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    scale = 0.03 # this value can be from 0 to 1 (0,1] to change the size of the text relative to the image
                    fontScale = min(w,h)/(25/scale)
                    
                    cv2.putText(img, lbls[0], (10, 40), font, fontScale, (0, 255, 0), 2)
                    
                    cv2.putText(img, lbls[1], (10, 70), font, fontScale, (0, 255, 0), 2)
                    
                    cv2.putText(img, lbls[2], (10, 100), font, fontScale, (0, 255, 0), 2)
                    
                    cv2.putText(img, lbls[3], (10, 130), font, fontScale, (0, 255, 0), 2)

                    cv2.putText(img, lbls[4], (10, 160), font, fontScale, (0, 255, 0), 2)

                    cv2.imwrite(output_dir_images+"slice"+str(SLICE_COUNT-(i))+".png", img)

        #either save things to the inference_pipeline or return them for the next class
        return {}


