import sys
import os
import time
import random

# from thop import profile
import torch
from torch.nn import utils
# from progress.bar import Bar
from collections import OrderedDict
from PIL import Image

from base.framework_factory import load_framework
from base.util import *
from base.data import get_loader, Test_Dataset
from test import test_model

torch.set_printoptions(precision=5)


def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
        print("loading on {}".format(net_name))
    else:
        print('Need model name!')
        return

    # Loading model
    config, model, optim, sche, model_loss, saver = load_framework(net_name)
    ave_batch = config['agg_batch'] // config['batch']

    # agg_batch: batch size for backwarding.
    # batch: batch size when loading to gpus. Decided by the GPU memory.
    print(sorted(config.items()))

    print(
        f"Training {config['model_name']} with {config['backbone']} backbone using {config['strategy']} strategy on GPU: {config['gpus']}.")

    # Loading datasets
    train_loader = get_loader(config)
    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)

    start_epoch = 1
    if config['resume']:
        saved_model = torch.load(config['weight'], map_location='cpu')
        if config['num_gpu'] > 1:
            model.module.load_state_dict(saved_model)
        else:
            model.load_state_dict(saved_model)

        start_epoch = int(config['weight'].split('_')[-1].split('.')[0]) + 1

    debug = config['debug']
    num_epoch = config['epoch']
    num_iter = len(train_loader)
    # ave_batch = config['ave_batch']
    trset = config['trset']
    model.zero_grad()
    for epoch in range(start_epoch, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()

        if debug:
            test_model(model, test_sets, config, epoch)

        # bar = Bar('{:10}-{:8} | epoch {:2}:'.format(net_name, config['sub'], epoch), max=num_iter)

        config['cur_epoch'] = epoch
        config['iter_per_epoch'] = num_iter
        st = time.time()
        optim.zero_grad()
        loss_count = 0
        batch_idx = 0
        # sche.step()
        for i, pack in enumerate(train_loader, start=1):
            current_iter = (epoch - 1) * num_iter + i
            total_iter = num_epoch * num_iter
            # print('iter: ', total_iter, current_iter)

            sche(optim, current_iter, total_iter, config)

            images, gts = pack
            images, gts = images.float().cuda(), gts.float().cuda()

            if config['multi']:
                #scales = [-1, 0, 1]
                scales = [-2, -1, 0, 1, 2]
                input_size = config['size']
                input_size += int(np.random.choice(scales, 1) * 64)
                #input_size += int(np.random.choice(scales, 1) * 32)
                images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')

            Y = model(images, 'train')

            for name, param in model.named_parameters():
                print(f"{name}: {param.requires_grad}")
            # 检查模型输出的 requires_grad
            print(f"Model output Y requires grad: {Y.requires_grad}")
            exit()
            loss = model_loss(Y, gts, config) / ave_batch
            loss_count += loss.data
            loss.backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                optim.zero_grad()
                batch_idx = 0

            lrs = ','.join([format(param['lr'], ".1e") for param in optim.param_groups])
            print('epoch {:2} | {:4}/{:4} | loss: {:1.6f}, LRs: [{}], time: {:1.3f}.'.format(epoch, i, num_iter,
                                                                                             float(loss_count / i), lrs,
                                                                                             time.time() - st))

        if epoch > num_epoch - 3:
            weight_path = os.path.join(config['weight_path'],
                                       '{}_{}_{}_{}.pth'.format(config['model_name'], config['backbone'], config['sub'],
                                                                epoch))
            torch.save(model.state_dict(), weight_path)
            test_model(model, test_sets, config, epoch)


if __name__ == "__main__":
    main()
