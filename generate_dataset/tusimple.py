import os
import os.path as ops
import shutil

import cv2
import numpy as np

def gen_train_sample(src_dir):
    """
    generate sample index file
    """

    with open('{:s}/train.txt'.format(src_dir), 'w') as file:

        for image_folder in os.listdir(src_dir):
            if image_folder.endswith('.txt'):
                continue
            
            print(image_folder)
            fr_dir = os.path.join(src_dir, image_folder)
            
            for se_path in os.listdir(fr_dir):
                image_dir = os.path.join(fr_dir, se_path)
                for image_name in os.listdir(image_dir):
                    if not image_name.endswith('.jpg'):
                        continue

                    image_path = ops.join(image_dir, image_name)

                    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
                    
                    print(image_path)

                    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    
                    info = '{:s}'.format(image_path)
                    file.write(info + '\n')
    return

def gen_val_sample(src_dir):
    """
    generate sample index file
    """

    with open('{:s}/val.txt'.format(src_dir), 'w') as file:

        for image_folder in os.listdir(src_dir):
            if image_folder.endswith('.txt'):
                continue
            
            print(image_folder)
            fr_dir = os.path.join(src_dir, image_folder)
            
            for se_path in os.listdir(fr_dir):
                image_dir = os.path.join(fr_dir, se_path)
                for image_name in os.listdir(image_dir):
                    if not image_name.endswith('.jpg'):
                        continue

                    image_path = ops.join(image_dir, image_name)

                    assert ops.exists(image_path), '{:s} not exist'.format(image_path)
                    
                    print(image_path)

                    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    
                    info = '{:s}'.format(image_path)
                    file.write(info + '\n')
    return

if __name__ == '__main__':
    
    train_dir = r'D:\Tusimple\clips'
    val_dir = r'D:\Tusimple\test_set\clips'
    gen_train_sample(train_dir)
    gen_val_sample(val_dir)
    
    