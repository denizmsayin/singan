from sys import argv

import torch

from utils import *

if __name__ == '__main__':
    if len(argv) != 4:
        print('usage: python upgrade_model.py model_file orig_img max_size')
        exit()

    _, model_path, img_path, max_size = argv
    max_size = int(max_size)

    tensor_image = load_image(img_path, max_size)
    save_dict = torch.load(model_path)
    save_dict['image'] = normed_tensor_to_np_image(tensor_image)

    save_dict['generators'] = [model.to('cpu') for model in save_dict['generators']]
    save_dict['critics'] = [model.to('cpu') for model in save_dict['critics']]

    save_dict['downsampling_mode'] = 'bicubic'
    if 'scaling_mode' in save_dict.keys():
        save_dict['upsampling_mode'] = save_dict['scaling_mode']
        del save_dict['scaling_mode']

    torch.save(save_dict, model_path)
