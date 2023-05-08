import argparse
import os
import pickle
import sys

import nibabel as nib
import numpy as np
import scipy
import SimpleITK as sitk
from scipy import ndimage as ndi


def loadNiiToArray(path):
    NiImg = nib.load(path)
    array = np.array(NiImg.dataobj)
    return array


def loadNiiWithSitk(path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(path)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)
    return array


def loadNiiImageWithSitk(path):
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(path)
    image = reader.Execute()
    # invert the image to be compatible with Nibabel
    image = sitk.Flip(image, [False, True, False])
    return image


def keep_masked_values(arr, mask):
    # Get the indices of the non-zero elements in the mask
    mask_indices = np.nonzero(mask)
    # Use the indices to select the corresponding elements from the array
    masked_values = arr[mask_indices]
    print(mask.shape)
    print(masked_values.shape)
    # Return the selected elements as a new array
    return masked_values


def get_stats(arr):
    # # Get the indices of the non-zero elements in the array
    # nonzero_indices = np.nonzero(arr)
    # # Use the indices to get the non-zero elements of the array
    # nonzero_elements = arr[nonzero_indices]

    nonzero_elements = arr

    # Calculate the stats for the non-zero elements
    max_val = np.max(nonzero_elements)
    min_val = np.min(nonzero_elements)
    mean_val = np.mean(nonzero_elements)
    median_val = np.median(nonzero_elements)
    std_val = np.std(nonzero_elements)
    variance_val = np.var(nonzero_elements)
    return max_val, min_val, mean_val, median_val, std_val, variance_val


def getMaskAnteriorAtrium(mask):
    erasePreAtriumMask = mask.copy()
    for sliceNum in range(mask.shape[-1]):
        mask2D = mask[:, :, sliceNum]
        itemindex = np.where(mask2D == 1)
        if itemindex[0].size > 0:
            row = itemindex[0][0]
            erasePreAtriumMask[:, :, sliceNum][:row, :] = 1
    return erasePreAtriumMask


"""
Function from
https://stackoverflow.com/questions/46310603/how-to-compute-convex-hull-image-volume-in-3d-numpy-arrays/46314485#46314485
"""


def fill_hull(image):
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img


def getClassBinaryMask(TSOutArray, classNum):
    binaryMask = np.zeros(TSOutArray.shape)
    binaryMask[TSOutArray == classNum] = 1
    return binaryMask


def loadNiftis(TSNiftiPath, imageNiftiPath):
    TSArray = loadNiiToArray(TSNiftiPath)
    scanArray = loadNiiToArray(imageNiftiPath)
    return TSArray, scanArray


# function to select one slice from 3D volume of SimpleITK image
def selectSlice(scanImage, zslice):
    size = list(scanImage.GetSize())
    size[2] = 0
    index = [0, 0, zslice]

    Extractor = sitk.ExtractImageFilter()
    Extractor.SetSize(size)
    Extractor.SetIndex(index)

    sliceImage = Extractor.Execute(scanImage)
    return sliceImage


# function to apply windowing
def windowing(sliceImage, center=400, width=400):
    windowMinimum = center - (width / 2)
    windowMaximum = center + (width / 2)
    img_255 = sitk.Cast(
        sitk.IntensityWindowing(
            sliceImage,
            windowMinimum=-windowMinimum,
            windowMaximum=windowMaximum,
            outputMinimum=0.0,
            outputMaximum=255.0,
        ),
        sitk.sitkUInt8,
    )
    return img_255


def selectSampleSlice(kidneyLMask, adRMask, scanImage):
    # Get the middle slice of the kidney mask from where there is the first 1 value to the last 1 value
    middleSlice = np.where(kidneyLMask.sum(axis=(0, 1)) > 0)[0][0] + int(
        (
            np.where(kidneyLMask.sum(axis=(0, 1)) > 0)[0][-1]
            - np.where(kidneyLMask.sum(axis=(0, 1)) > 0)[0][0]
        )
        / 2
    )
    print("Middle slice: ", middleSlice)
    # make middleSlice int
    middleSlice = int(middleSlice)
    # select one slice using simple itk
    sliceImageK = selectSlice(scanImage, middleSlice)

    # Get the middle slice of the addrenal mask from where there is the first 1 value to the last 1 value
    middleSlice = np.where(adRMask.sum(axis=(0, 1)) > 0)[0][0] + int(
        (
            np.where(adRMask.sum(axis=(0, 1)) > 0)[0][-1]
            - np.where(adRMask.sum(axis=(0, 1)) > 0)[0][0]
        )
        / 2
    )
    print("Middle slice: ", middleSlice)
    # make middleSlice int
    middleSlice = int(middleSlice)
    # select one slice using simple itk
    sliceImageA = selectSlice(scanImage, middleSlice)

    sliceImageK = windowing(sliceImageK)
    sliceImageA = windowing(sliceImageA)

    return sliceImageK, sliceImageA


def getFeatures(TSArray, scanArray):
    aortaMask = getClassBinaryMask(TSArray, 7)
    IVCMask = getClassBinaryMask(TSArray, 8)
    portalMask = getClassBinaryMask(TSArray, 9)
    print(np.unique(portalMask))
    atriumMask = getClassBinaryMask(TSArray, 45)
    kidneyLMask = getClassBinaryMask(TSArray, 3)
    kidneyRMask = getClassBinaryMask(TSArray, 2)
    adRMask = getClassBinaryMask(TSArray, 11)

    # Remove toraccic aorta adn IVC from aorta and IVC masks
    anteriorAtriumMask = getMaskAnteriorAtrium(atriumMask)
    aortaMask = aortaMask * (anteriorAtriumMask == 0)
    IVCMask = IVCMask * (anteriorAtriumMask == 0)

    # Erode vessels to get only the center of the vessels
    struct2 = np.ones((3, 3, 3))
    aortaMaskEroded = ndi.binary_erosion(aortaMask, structure=struct2).astype(aortaMask.dtype)
    IVCMaskEroded = ndi.binary_erosion(IVCMask, structure=struct2).astype(IVCMask.dtype)

    struct3 = np.ones((1, 1, 1))
    portalMaskEroded = ndi.binary_erosion(portalMask, structure=struct3).astype(portalMask.dtype)
    # If portalMaskEroded has less then 500 values, use the original portalMask
    if np.count_nonzero(portalMaskEroded) < 500:
        portalMaskEroded = portalMask
        
    # Get masked values from scan
    aortaArray = keep_masked_values(scanArray, aortaMaskEroded)
    IVCArray = keep_masked_values(scanArray, IVCMaskEroded)
    portalArray = keep_masked_values(scanArray, portalMaskEroded)
    kidneyLArray = keep_masked_values(scanArray, kidneyLMask)
    kidneyRArray = keep_masked_values(scanArray, kidneyRMask)

    """Put this on a separate function and return only the pelvis arrays"""
    # process the Renal Pelvis masks from the Kidney masks
    # create the convex hull of the Left Kidney
    kidneyLHull = fill_hull(kidneyLMask)
    # exclude the Left Kidney mask from the Left Convex Hull
    kidneyLHull = kidneyLHull * (kidneyLMask == 0)
    # erode the kidneyHull to remove the edges
    struct = np.ones((3, 3, 3))
    kidneyLHull = ndi.binary_erosion(kidneyLHull, structure=struct).astype(kidneyLHull.dtype)
    # keep the values of the scanArray that are in the Left Convex Hull
    pelvisLArray = keep_masked_values(scanArray, kidneyLHull)

    # create the convex hull of the Right Kidney
    kidneyRHull = fill_hull(kidneyRMask)
    # exclude the Right Kidney mask from the Right Convex Hull
    kidneyRHull = kidneyRHull * (kidneyRMask == 0)
    # erode the kidneyHull to remove the edges
    struct = np.ones((3, 3, 3))
    kidneyRHull = ndi.binary_erosion(kidneyRHull, structure=struct).astype(kidneyRHull.dtype)
    # keep the values of the scanArray that are in the Right Convex Hull
    pelvisRArray = keep_masked_values(scanArray, kidneyRHull)

    # Get the stats
    # Get the stats for the aortaArray
    (
        aorta_max_val,
        aorta_min_val,
        aorta_mean_val,
        aorta_median_val,
        aorta_std_val,
        aorta_variance_val,
    ) = get_stats(aortaArray)

    # Get the stats for the IVCArray
    (
        IVC_max_val,
        IVC_min_val,
        IVC_mean_val,
        IVC_median_val,
        IVC_std_val,
        IVC_variance_val,
    ) = get_stats(IVCArray)

    # Get the stats for the portalArray
    (
        portal_max_val,
        portal_min_val,
        portal_mean_val,
        portal_median_val,
        portal_std_val,
        portal_variance_val,
    ) = get_stats(portalArray)

    # Get the stats for the kidneyLArray and kidneyRArray
    (
        kidneyL_max_val,
        kidneyL_min_val,
        kidneyL_mean_val,
        kidneyL_median_val,
        kidneyL_std_val,
        kidneyL_variance_val,
    ) = get_stats(kidneyLArray)
    (
        kidneyR_max_val,
        kidneyR_min_val,
        kidneyR_mean_val,
        kidneyR_median_val,
        kidneyR_std_val,
        kidneyR_variance_val,
    ) = get_stats(kidneyRArray)

    (
        pelvisL_max_val,
        pelvisL_min_val,
        pelvisL_mean_val,
        pelvisL_median_val,
        pelvisL_std_val,
        pelvisL_variance_val,
    ) = get_stats(pelvisLArray)
    (
        pelvisR_max_val,
        pelvisR_min_val,
        pelvisR_mean_val,
        pelvisR_median_val,
        pelvisR_std_val,
        pelvisR_variance_val,
    ) = get_stats(pelvisRArray)

    # create three new columns for the decision tree
    # aorta - porta, Max min and mean columns
    aorta_porta_max = aorta_max_val - portal_max_val
    aorta_porta_min = aorta_min_val - portal_min_val
    aorta_porta_mean = aorta_mean_val - portal_mean_val

    # aorta - IVC, Max min and mean columns
    aorta_IVC_max = aorta_max_val - IVC_max_val
    aorta_IVC_min = aorta_min_val - IVC_min_val
    aorta_IVC_mean = aorta_mean_val - IVC_mean_val

    # Save stats in CSV:
    # Create a list to store the stats
    stats = []
    # Add the stats for the aortaArray to the list
    stats.extend(
        [
            aorta_max_val,
            aorta_min_val,
            aorta_mean_val,
            aorta_median_val,
            aorta_std_val,
            aorta_variance_val,
        ]
    )
    # Add the stats for the IVCArray to the list
    stats.extend(
        [IVC_max_val, IVC_min_val, IVC_mean_val, IVC_median_val, IVC_std_val, IVC_variance_val]
    )
    # Add the stats for the portalArray to the list
    stats.extend(
        [
            portal_max_val,
            portal_min_val,
            portal_mean_val,
            portal_median_val,
            portal_std_val,
            portal_variance_val,
        ]
    )
    # Add the stats for the kidneyLArray and kidneyRArray to the list
    stats.extend(
        [
            kidneyL_max_val,
            kidneyL_min_val,
            kidneyL_mean_val,
            kidneyL_median_val,
            kidneyL_std_val,
            kidneyL_variance_val,
        ]
    )
    stats.extend(
        [
            kidneyR_max_val,
            kidneyR_min_val,
            kidneyR_mean_val,
            kidneyR_median_val,
            kidneyR_std_val,
            kidneyR_variance_val,
        ]
    )
    # Add the stats for the kidneyLHull and kidneyRHull to the list
    stats.extend(
        [
            pelvisL_max_val,
            pelvisL_min_val,
            pelvisL_mean_val,
            pelvisL_median_val,
            pelvisL_std_val,
            pelvisL_variance_val,
        ]
    )
    stats.extend(
        [
            pelvisR_max_val,
            pelvisR_min_val,
            pelvisR_mean_val,
            pelvisR_median_val,
            pelvisR_std_val,
            pelvisR_variance_val,
        ]
    )

    stats.extend(
        [
            aorta_porta_max,
            aorta_porta_min,
            aorta_porta_mean,
            aorta_IVC_max,
            aorta_IVC_min,
            aorta_IVC_mean,
        ]
    )

    return stats, kidneyLMask, adRMask


def loadModel():
    c2cPath = os.path.dirname(sys.path[0])
    filename = os.path.join(c2cPath, "comp2comp", "contrast_phase", "xgboost.pkl")
    model = pickle.load(open(filename, "rb"))

    return model


def predict_phase(TS_path, scan_path, outputPath=None, save_sample=False):
    TS_array, image_array = loadNiftis(TS_path, scan_path)
    model = loadModel()
    # TS_array, image_array = loadNiftis(TS_output_nifti_path, image_nifti_path)
    featureArray, kidneyLMask, adRMask = getFeatures(TS_array, image_array)
    y_pred = model.predict([featureArray])

    if y_pred == 0:
        pred_phase = "non-contrast"
    if y_pred == 1:
        pred_phase = "arterial"
    if y_pred == 2:
        pred_phase = "venous"
    if y_pred == 3:
        pred_phase = "delayed"

    output_path_metrics = os.path.join(outputPath, "metrics")
    if not os.path.exists(output_path_metrics):
        os.makedirs(output_path_metrics)
    outputTxt = os.path.join(output_path_metrics, "phase_prediction.txt")
    with open(outputTxt, "w") as text_file:
        text_file.write(pred_phase)
    print(pred_phase)

    output_path_images = os.path.join(outputPath, "images")
    if not os.path.exists(output_path_images):
        os.makedirs(output_path_images)
    scanImage = loadNiiImageWithSitk(scan_path)
    sliceImageK, sliceImageA = selectSampleSlice(kidneyLMask, adRMask, scanImage)
    outJpgK = os.path.join(output_path_images, "sampleSliceKidney.png")
    sitk.WriteImage(sliceImageK, outJpgK)
    outJpgA = os.path.join(output_path_images, "sampleSliceAdrenal.png")
    sitk.WriteImage(sliceImageA, outJpgA)


if __name__ == "__main__":
    # parse arguments optional
    parser = argparse.ArgumentParser()
    parser.add_argument("--TS_path", type=str, required=True, help="Input image")
    parser.add_argument("--scan_path", type=str, required=True, help="Input image")
    parser.add_argument(
        "--output_dir", type=str, required=False, help="Output .txt prediction", default=None
    )
    parser.add_argument(
        "--save_sample", type=bool, required=False, help="Save jpeg sample ", default=False
    )
    args = parser.parse_args()
    predict_phase(args.TS_path, args.scan_path, args.output_dir, args.save_sample)
