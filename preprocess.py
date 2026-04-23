import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm
from process_retouch import preprocess_oct_images


def safe_remove(path):
    if os.path.islink(path) or os.path.isfile(path):
        try:
            os.unlink(path)
        except Exception:
            os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def img_crop(im, height, width, X, Y, D):
    # crop image into a square with size D*D
    h, w = im.shape[:2]
    # Some images may have size slightly different from the metadata; resize to expected size for consistency
    if (h, w) != (height, width):
        print(f'WARNING: size mismatch, got {(h, w)} but expected {(height, width)}; resizing to expected size')
        # cv2.resize expects (width, height)
        im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
        h, w = im.shape[:2]
    
    # Handle both color images (3 channels) and grayscale/label images (2D or 1 channel)
    is_grayscale = len(im.shape) == 2
    if is_grayscale:
        square_im = np.zeros((D, D), dtype=im.dtype)
    else:
        square_im = np.zeros((D, D, im.shape[2]), dtype=im.dtype)
    
    dh = min(h, D + Y) - max(0, Y)
    dw = min(w, D + X) - max(0, X)

    square_h_start = max(0, -Y)
    square_h_end = min(D, square_h_start + dh)
    square_w_start = max(0, -X)
    square_w_end = min(D, square_w_start + dw)

    raw_h_start = max(0, Y)
    raw_h_end = min(h, raw_h_start + dh)
    raw_w_start = max(0, X)
    raw_w_end = min(w, raw_w_start + dw)

    if is_grayscale:
        square_im[square_h_start:square_h_end, square_w_start:square_w_end] = im[raw_h_start:raw_h_end, raw_w_start:raw_w_end]
    else:
        square_im[square_h_start:square_h_end, square_w_start:square_w_end, :] = im[raw_h_start:raw_h_end, raw_w_start:raw_w_end, :]
    
    circle_mask = np.zeros((D, D), dtype=np.uint8)
    cv2.circle(circle_mask, (int(D/2), int(D/2)), int(D/2), 255, -1)
    square_im[circle_mask==0] = 0
    return square_im


def main(img_root, out_root, info_path, copy_unprocessed_data):
    datasets_without_preprocess = ['OCTDL', 'NEH', 'OCTID', 'UCSD', 'TOP', 'MMC-AMD', os.path.join('DeepDRiD', 'ultra-widefield_images')]
   
    
    # for symbolic link
    img_root = os.path.abspath(img_root)
    out_root = os.path.abspath(out_root)

    # process RETOUCH (skip if already processed)
    retouch_out = os.path.join(out_root, 'RETOUCH', 'pre_processed')
    if os.path.exists(retouch_out) and len(os.listdir(retouch_out)) > 0:
        print('RETOUCH pre_processed already exists and is non-empty, skipping RETOUCH processing')
    else:
        preprocess_oct_images(os.path.join(img_root, 'RETOUCH'), retouch_out)
    
    # process CFP
    with open(info_path) as fin:
        process_info = json.load(fin)
    for dataset in process_info:
        print('processing', dataset)
        # dataset-level done flag
        dataset_done_flag = os.path.join(out_root, f"{dataset.replace(os.sep, '_')}.done")

        # 1) If a .done flag exists, skip this dataset entirely
        if os.path.exists(dataset_done_flag):
            print(f'dataset {dataset} already marked as processed, skipping')
            continue

        # 2) If all expected outputs already exist, auto-mark as done and skip
        all_outputs_exist = True
        for img_name in process_info[dataset]:
            out_path = os.path.join(out_root, img_name)
            if not os.path.exists(out_path):
                all_outputs_exist = False
                break
        if all_outputs_exist:
            print(f'all outputs already exist for dataset {dataset}, marking as done and skipping')
            with open(dataset_done_flag, 'w') as _f:
                _f.write('auto-detected processed\n')
            continue

        # 3) Otherwise, process images in this dataset
        for img_name in tqdm(process_info[dataset]):
            img_path = os.path.join(img_root, img_name)
            print(img_path)
            out_path = os.path.join(out_root, img_name)
            out_dir = os.path.dirname(out_path)
            # if output already exists, skip to support resume
            if os.path.exists(out_path):
                # preserve earlier output
                # uncomment the following to overwrite instead
                # safe_remove(out_path)
                # continue to next image
                print(f'output exists, skipping {out_path}')
                continue
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
                
            rotate = process_info[dataset][img_name]['rotate']
            crop = process_info[dataset][img_name]['crop']    

            # images without preprocess
            if crop is None and not rotate:
                if copy_unprocessed_data:
                    shutil.copy(img_path, out_path)
                else:
                    if os.path.exists(out_path) or os.path.islink(out_path):
                        safe_remove(out_path)
                    os.symlink(img_path, out_path)
            
            # images with preprocess
            else:
                im = cv2.imread(img_path)
                if im is None:
                    print(f'WARNING: failed to read image {img_path}, skipping')
                    continue
                if crop is not None:
                    height, width, X, Y, D = crop
                    im = img_crop(im, height, width, X, Y, D)
                if rotate:
                    im = cv2.rotate(im, cv2.ROTATE_180)
                    
                cv2.imwrite(out_path, im)
                
                # Process corresponding label files for lesion segmentation task
                if 'lesion_segmentation' in img_name and '/image/' in img_name:
                    lesion_types = ['EX', 'HE', 'MA', 'SE']
                    for lesion_type in lesion_types:
                        # Construct label path from image path
                        label_img_path = img_path.replace('/image/', f'/label/{lesion_type}/')
                        label_img_path = label_img_path.replace('.jpg', '.tif')
                        
                        if os.path.exists(label_img_path):
                            label_out_path = out_path.replace('/image/', f'/label/{lesion_type}/')
                            label_out_path = label_out_path.replace('.jpg', '.tif')
                            label_out_dir = os.path.dirname(label_out_path)
                            
                            # Skip if label output already exists
                            if os.path.exists(label_out_path):
                                print(f'label output exists, skipping {label_out_path}')
                                continue
                            
                            if not os.path.exists(label_out_dir):
                                os.makedirs(label_out_dir, exist_ok=True)
                            
                            # Read label as grayscale (segmentation mask)
                            label = cv2.imread(label_img_path, cv2.IMREAD_UNCHANGED)
                            
                            # Apply the same transformations as the image
                            if crop is not None:
                                label = img_crop(label, height, width, X, Y, D)
                            if rotate:
                                label = cv2.rotate(label, cv2.ROTATE_180)
                            
                            cv2.imwrite(label_out_path, label)

        # 4) After successfully iterating this dataset, mark it as done
        with open(dataset_done_flag, 'w') as _f:
            _f.write('processed\n')
                    
    for dataset in datasets_without_preprocess:
        print('processing', dataset)
        dest = os.path.join(out_root, dataset)
        if os.path.exists(dest) or os.path.islink(dest):
            print('path {} already exists, overwriting'.format(dest))
            safe_remove(dest)
        if copy_unprocessed_data:
            shutil.copytree(os.path.join(img_root, dataset), dest, dirs_exist_ok=True)
        else:
            os.symlink(os.path.join(img_root, dataset), dest)
            
if __name__ == '__main__':
    img_root = 'datasets'
    out_root = 'datasets_preprocessed'

    # The pre-calculated crop and rotate information for CFP datasets.
    # For Windows users, you may need to replace '/' with '\\' in the image paths.
    info_path = 'preprocess_info.json'

    # For unprocessed images, whether to copy the images to the output directory or just create a symbolic link
    copy_unprocessed_data = False
    main(img_root, out_root, info_path, copy_unprocessed_data)