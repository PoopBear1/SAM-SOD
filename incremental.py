import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# 假设 YourModelWithLoRA 是一个包含LoRA参数的预训练模型
from methods.adalora import Network as LoRA
from base.framework_factory import load_framework


def determine_n_class(saved_model_weights):
    for name, weight in saved_model_weights.items():
        if 'mask_tokens' in name:
            current_n_class = weight.shape[0]
            return current_n_class + 1


def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return

    saved_model_weights = torch.load('./weight/adalora/resnet50/base/adalora_resnet50_base_8.pth', map_location='cpu')

    incremented_n_class = determine_n_class(saved_model_weights)
    config, model, _, _, _, saver = load_framework(net_name, incremented_n_class)

    sam_model_state_dict = model.state_dict()

    # check if n_class has increased
    change_n_class = model.sam.mask_decoder.mask_tokens.weight.shape[0]
    print("after incremental, the class is: ", change_n_class)

    for name, param in saved_model_weights.items():
        if name in sam_model_state_dict and sam_model_state_dict[name].shape == param.shape:
            sam_model_state_dict[name].copy_(param)


    model.load_state_dict(sam_model_state_dict, strict=False)

    for param in model.sam.image_encoder.parameters():
        param.requires_grad = False

    exit()


if __name__ == '__main__':
    main()
