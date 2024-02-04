import torch
from torch import nn
from methods.segment_anything import sam_model_registry, SamPredictor

custom_config = {'base': {'strategy': 'base_adam',
                          'batch': 8,
                          },
                 'customized': {'--ckpt_path': {'type': str, 'default': './ckpts/sam_vit_b_01ec64.pth'},
                                '--model_type': {'type': str, 'default': 'vit_b'},
                                '--train_encoder': {'action': 'store_true'},
                                '--train_decoder': {'action': 'store_true'},
                                '--train_prompt': {'action': 'store_true'},
                                },
                 }

# custom_config = {'base': {'strategy': 'base_adam',
#                           'batch': 8,
#                           },
#                  'customized': {'--ckpt_path': {'type': str, 'default': './ckpts/sam_vit_h_4b8939.pth'},
#                                 '--model_type': {'type': str, 'default': 'vit_h'},
#                                 '--train_encoder': {'action': 'store_true'},
#                                 '--train_decoder': {'action': 'store_true'},
#                                 '--train_prompt': {'action': 'store_true'},
#                                 },
#                  }

class Network(nn.Module):
    def __init__(self, config, encoder, feat):
        # encoder: backbone, forward function output 5 encoder features. details in methods/base/model.py
        # feat: length of encoder features. e.g.: VGG:[64, 128, 256, 512, 512]; Resnet:[64, 256, 512, 1024, 2048]
        super(Network, self).__init__()
        sam_checkpoint = config['ckpt_path']
        model_type = config['model_type']
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        if not config['train_encoder']:
            for param in self.sam.image_encoder.parameters():
                param.requires_grad = False
        if not config['train_decoder']:
            for param in self.sam.mask_decoder.parameters():
                param.requires_grad = False
        if not config['train_prompt']:
            for param in self.sam.prompt_encoder.parameters():
                param.requires_grad = False

    def forward(self, x, phase='test'):
        batched_input = x
        image_size = batched_input.shape[-1]
        out = self.sam(batched_input, multimask_output=False, image_size=image_size)
        out_dict = {'sal': out['masks'], 'final': out['masks']}
        return out_dict
