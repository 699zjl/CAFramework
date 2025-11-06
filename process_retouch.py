import os
import SimpleITK as sitk
import numpy as np

import pandas as pd
from PIL import Image

"""
This script is from the Internet (probably Kaggle) and we are not sure about the author.
"""

def load_oct_image(filename):
    """
    loads an .mhd file using simple_itk
    :param filename: name of the image to be loaded
    :return: int32 3D image with voxels range 0-255
    """

    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    ct_scan = ct_scan.astype(np.int32)
    num_slices = ct_scan.shape[0]
    oct_scan_ret = np.zeros(ct_scan.shape, dtype=np.int32)
    if 'Cirrus' in filename:
        # range 0-255
        oct_scan_ret = ct_scan.astype(np.int32)
    elif 'Spectralis' in filename:
        # range 0-2**16
        oct_scan_ret = (ct_scan.astype(np.float32) / (2 ** 16) * 255.).astype(np.int32)
    elif 'Topcon' in filename:
        # range 0-255
        oct_scan_ret = ct_scan.astype(np.int32)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return oct_scan_ret, origin, spacing


def load_oct_seg(filename):
    """
    loads an .mhd file using simple_itk
    :param filename: 
    :return: 
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    oct_scan = sitk.GetArrayFromImage(itkimage)
    oct_scan = oct_scan.astype(np.int8)
    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return oct_scan, origin, spacing

IRF_CODE = 1
SRF_CODE = 2
PED_CODE = 3


def preprocess_oct_images(data_root, out_root):
    # Prepare reference image for histogram matching (randomly selected from Spectralis dataset)
    #filepath_r = '/home/truwan/DATA/retouch/Spectralis/7501081e3e7577af524c6f7703d8d538/oct.mhd'
    #oct_r, _, _ = load_oct_image(filepath_r)
    count = 0
    image_names = list()
    for subdir, dirs, files in os.walk(data_root):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith("reference.mhd"):
                image_name = filepath.split('/')[-2]
                vendor = filepath.split('/')[-3]
                img, _, _ = load_oct_seg(filepath)
                num_slices = img.shape[0]
                for slice_num in range(0, num_slices):
                    im_slice = img[slice_num, :, :]
                    image_names.append([image_name, vendor, subdir, slice_num, int(np.any(im_slice == IRF_CODE)),
                                        int(np.any(im_slice == SRF_CODE)), int(np.any(im_slice == PED_CODE))])
                    im_slice = Image.fromarray(im_slice.astype(np.int32)*50, mode='I').convert('RGBA')
                    #os.makedirs('./pre_processed/oct_masks/', exist_ok=True)
                    os.makedirs(os.path.join(out_root, 'oct_masks'), exist_ok=True)
                    #save_name = './pre_processed/oct_masks/' + vendor + '_' + image_name + '_' + str(slice_num).zfill(3) + '.png'
                    save_name = os.path.join(out_root, 'oct_masks', vendor + '_' + image_name + '_' + str(slice_num).zfill(3) + '.png')
                    im_slice.save(save_name)

            elif filepath.endswith("oct.mhd"):
                image_name = filepath.split('/')[-2]
                vendor = filepath.split('/')[-3]
                img, _, _ = load_oct_image(filepath)
                num_slices = img.shape[0]
                
                for slice_num in range(0, num_slices):
                    im_slice = img[slice_num, :, :].astype(np.int32)
                    im_slice = Image.fromarray(im_slice, mode='I').convert('RGBA')
                    #os.makedirs('./pre_processed/oct_imgs/', exist_ok=True)
                    os.makedirs(os.path.join(out_root, 'oct_imgs'), exist_ok=True)
                    #save_name = './pre_processed/oct_imgs/' + vendor + '_' + image_name + '_' + str(slice_num).zfill(3) + '.png'
                    save_name = os.path.join(out_root, 'oct_imgs', vendor + '_' + image_name + '_' + str(slice_num).zfill(3) + '.png')
                    im_slice.save(save_name)

    col_names = ['image_name', 'vendor', 'root', 'slice', 'is_IRF', 'is_SRF', 'is_PED']
    df = pd.DataFrame(image_names, columns=col_names)
    #df.to_csv('./pre_processed/slice_gt.csv', index=False)
    df.to_csv(os.path.join(out_root, 'slice_gt.csv'), index=False)


    
