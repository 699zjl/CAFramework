import os
import shutil
import json
import cv2
import numpy as np
from tqdm import tqdm

def img_crop(im, height, width, X, Y, D):
    # crop image into a square with size D*D
    h, w = im.shape[:2]
    assert (h, w) == (height, width), (h, w, height, width)
    
    square_im = np.zeros((D, D, 3), dtype=np.uint8)
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

    square_im[square_h_start:square_h_end, square_w_start:square_w_end, :] = im[raw_h_start:raw_h_end, raw_w_start:raw_w_end, :]
    
    circle_mask = np.zeros((D, D), dtype=np.uint8)
    cv2.circle(circle_mask, (int(D/2), int(D/2)), int(D/2), 255, -1)
    square_im[circle_mask==0] = 0
    return square_im


def main(img_root, out_root, info_path, copy_unprocessed_data):
    datasets_without_preprocess = ['OCTDL', 'NEH', 'OCTID', 'UCSD', 'RETOUCH', 'TOP', 'MMC-AMD', os.path.join('DeepDRiD', 'ultra-widefield_images')]

    # for symbolic link
    img_root = os.path.abspath(img_root)
    out_root = os.path.abspath(out_root)

    with open(info_path) as fin:
        process_info = json.load(fin)

    for dataset in process_info:
        
        print('processing', dataset)
        for img_name in tqdm(process_info[dataset]):
            img_path = os.path.join(img_root, img_name)
            print(img_path)
            out_path = os.path.join(out_root, img_name)
            out_dir = os.path.dirname(out_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
                
            rotate = process_info[dataset][img_name]['rotate']
            crop = process_info[dataset][img_name]['crop']    

            # images without preprocess
            if crop is None and not rotate:
                if copy_unprocessed_data:
                    shutil.copy(img_path, out_path)
                else:
                    os.symlink(img_path, out_path)
            
            # images with preprocess
            else:
                im = cv2.imread(img_path)
                if crop is not None:
                    height, width, X, Y, D = crop
                    im = img_crop(im, height, width, X, Y, D)
                if rotate:
                    im = cv2.rotate(im, cv2.ROTATE_180)
                    
                cv2.imwrite(out_path, im)
                    

    for dataset in datasets_without_preprocess:
        print('processing', dataset)
        if os.path.exists(os.path.join(out_root, dataset)):
            print('path {} already exists, skip'.format(os.path.join(out_root, dataset)))
            continue
        if copy_unprocessed_data:
            shutil.copytree(os.path.join(img_root, dataset), os.path.join(out_root, dataset))
        else:
            os.symlink(os.path.join(img_root, dataset), os.path.join(out_root, dataset))
            
if __name__ == '__main__':
    img_root = 'datasets'
    out_root = 'datasets_preprocessed'

    # The pre-calculated crop and rotate information for CFP datasets.
    # For Windows users, you may need to replace '/' with '\\' in the image paths.
    info_path = 'preprocess_info.json'

    # For unprocessed images, whether to copy the images to the output directory or just create a symbolic link
    copy_unprocessed_data = False
    main(img_root, out_root, info_path, copy_unprocessed_data)