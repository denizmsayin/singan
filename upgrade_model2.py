from sys import argv

import torch

from utils import *

if __name__ == '__main__':
    if len(argv) < 2:
        print('usage: python upgrade_model.py model_files')
        exit()

    _, *model_paths = argv

    for model_path in model_paths:
        save_dict = torch.load(model_path)

        save_dict['generators'] = [model.to('cpu') for model in save_dict['generators']]
        save_dict['critics'] = [model.to('cpu') for model in save_dict['critics']]

        if 'scaling_factor' in save_dict.keys():
            save_dict['upsampling_factor'] = save_dict['scaling_factor']
            del save_dict['scaling_factor']

        torch.save(save_dict, model_path)

