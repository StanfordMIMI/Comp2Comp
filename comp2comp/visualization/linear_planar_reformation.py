"""
@author: louisblankemeier
"""

import numpy as np


def linear_planar_reformation(
    medical_volume: np.ndarray, segmentation: np.ndarray, centroids, dimension="axial"
):
    if dimension == "sagittal" or dimension == "coronal":
        centroids = sorted(centroids, key=lambda x: x[2])
    elif dimension == "axial":
        centroids = sorted(centroids, key=lambda x: x[0])

    centroids = [(int(x[0]), int(x[1]), int(x[2])) for x in centroids]
    sagittal_centroids = [centroids[i][0] for i in range(0, len(centroids))]
    coronal_centroids = [centroids[i][1] for i in range(0, len(centroids))]
    axial_centroids = [centroids[i][2] for i in range(0, len(centroids))]

    sagittal_vals, coronal_vals, axial_vals = [], [], []

    if dimension == "sagittal":
        sagittal_vals = [sagittal_centroids[0]] * axial_centroids[0]

    if dimension == "coronal":
        coronal_vals = [coronal_centroids[0]] * axial_centroids[0]

    if dimension == "axial":
        axial_vals = [axial_centroids[0]] * sagittal_centroids[0]

    for i in range(1, len(axial_centroids)):
        if dimension == "sagittal" or dimension == "coronal":
            num = axial_centroids[i] - axial_centroids[i - 1]
        elif dimension == "axial":
            num = sagittal_centroids[i] - sagittal_centroids[i - 1]

        if dimension == "sagittal":
            interp = list(np.linspace(sagittal_centroids[i - 1], sagittal_centroids[i], num=num))
            sagittal_vals.extend(interp)

        if dimension == "coronal":
            interp = list(np.linspace(coronal_centroids[i - 1], coronal_centroids[i], num=num))
            coronal_vals.extend(interp)

        if dimension == "axial":
            interp = list(np.linspace(axial_centroids[i - 1], axial_centroids[i], num=num))
            axial_vals.extend(interp)

    if dimension == "sagittal":
        sagittal_vals.extend(
            [sagittal_centroids[-1]] * (medical_volume.shape[2] - len(sagittal_vals))
        )
        sagittal_vals = np.array(sagittal_vals)
        sagittal_vals = sagittal_vals.astype(int)

    if dimension == "coronal":
        coronal_vals.extend([coronal_centroids[-1]] * (medical_volume.shape[2] - len(coronal_vals)))
        coronal_vals = np.array(coronal_vals)
        coronal_vals = coronal_vals.astype(int)

    if dimension == "axial":
        axial_vals.extend([axial_centroids[-1]] * (medical_volume.shape[0] - len(axial_vals)))
        axial_vals = np.array(axial_vals)
        axial_vals = axial_vals.astype(int)

    if dimension == "sagittal":
        sagittal_image = medical_volume[sagittal_vals, :, range(len(sagittal_vals))]
        sagittal_label = segmentation[sagittal_vals, :, range(len(sagittal_vals))]

    if dimension == "coronal":
        coronal_image = medical_volume[:, coronal_vals, range(len(coronal_vals))]
        coronal_label = segmentation[:, coronal_vals, range(len(coronal_vals))]

    if dimension == "axial":
        axial_image = medical_volume[range(len(axial_vals)), :, axial_vals]
        axial_label = segmentation[range(len(axial_vals)), :, axial_vals]

    if dimension == "sagittal":
        return sagittal_image, sagittal_label

    if dimension == "coronal":
        return coronal_image, coronal_label

    if dimension == "axial":
        return axial_image, axial_label
