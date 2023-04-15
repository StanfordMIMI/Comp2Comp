import os
import numpy as np
import pydicom
from PIL import Image
from pathlib import Path
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

def to_dicom(input, output_path, plane='axial'):
    # if input is string or path
    if isinstance(input, str) or isinstance(input, Path):
        png_path = input
        dicom_path = os.path.join(output_path, os.path.basename(png_path).replace('.png', '.dcm'))
        image = Image.open(png_path)
        image_array = np.array(image)
        image_array = image_array[:, :, :3]
    else:
        image_array = input

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
    meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

    ds = Dataset()
    ds.file_meta = meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.PatientName = "John Doe"
    ds.PatientID = "123456"
    ds.Modality = "OT"
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PhotometricInterpretation = "RGB"
    ds.PixelRepresentation = 0
    ds.Rows = image_array.shape[0]
    ds.Columns = image_array.shape[1]
    ds.SamplesPerPixel = 3
    ds.PlanarConfiguration = 0

    if plane.lower() == 'axial':
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    elif plane.lower() == 'sagittal':
        ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
    elif plane.lower() == 'coronal':
        ds.ImageOrientationPatient = [1, 0, 0, 0, 0, -1]
    else:
        raise ValueError("Invalid plane value. Must be 'axial', 'sagittal', or 'coronal'.")

    ds.PixelData = image_array.tobytes()
    pydicom.filewriter.write_file(dicom_path, ds, write_like_original=False)

# Example usage
if __name__ == '__main__':
    png_path = '../../figures/spine_example.png'
    output_path = './'
    plane = 'sagittal'
    png_to_dicom(png_path, output_path, plane)
