import sys
import time

from loralib import RankAllocator, compute_orth_regu

from base.data import get_loader
from base.framework_factory import load_framework
from base.util import *
from torch.nn import utils


def determine_n_class(saved_model_weights):
    for name, weight in saved_model_weights.items():
        if 'mask_tokens' in name:
            current_n_class = weight.shape[0]
            return current_n_class + 1


def train(config, model, optim, sche, model_loss):
    train_loader = get_loader(config)
    start_epoch = 1
    ave_batch = config['agg_batch'] // config['batch']

    if config['resume']:
        print("resume for training")
        saved_model = torch.load(config['weight'], map_location='cpu')
        if config['num_gpu'] > 1:
            model.module.load_state_dict(saved_model)
        else:
            model.load_state_dict(saved_model)

        start_epoch = int(config['weight'].split('_')[-1].split('.')[0]) + 1
    else:
        print("training from scratch")

    num_epoch = config['epoch']
    num_iter = len(train_loader)

    # Initialize the RankAllocator
    print("num_epoch: {}, num_iter:{}".format(num_epoch, num_iter))
    init_warmup = int(num_iter / ave_batch)
    final_warmup = int((3 * num_iter) / ave_batch)
    total_step = int((num_epoch * num_iter) / ave_batch)

    rankallocator = RankAllocator(
        model, lora_r=config['rank'], target_rank=config['target_rank'],
        init_warmup=init_warmup, final_warmup=final_warmup, mask_interval=10,
        total_step=total_step, beta1=0.85, beta2=0.85,
    )

    model.zero_grad()
    global_step = 0
    for epoch in range(start_epoch, num_epoch + 1):
        model.train()
        torch.cuda.empty_cache()

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
            sche(optim, current_iter, total_iter, config)
            images, gts = pack
            images, gts = images.float().cuda(), gts.float().cuda()

            if config['multi']:
                scales = [-2, -1, 0, 1, 2]
                input_size = config['size']
                input_size += int(np.random.choice(scales, 1) * 64)
                # images = F.upsample(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                # gts = F.upsample(gts, size=(input_size, input_size), mode='nearest')
                images = F.interpolate(images, size=(input_size, input_size), mode='bilinear', align_corners=True)
                gts = gts.unsqueeze(1)  # 在第1维（C维）处添加一个大小为1的新维度
                gts = F.interpolate(gts, size=(input_size, input_size), mode='nearest')

            Y = model(images, 'train')
            loss = model_loss(Y, gts, config) / ave_batch
            loss_count += loss.data
            (loss + (compute_orth_regu(model, regu_weight=0.1) / ave_batch)).backward()

            batch_idx += 1
            if batch_idx == ave_batch:
                if config['clip_gradient']:
                    utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
                optim.step()
                rankallocator.update_and_mask(model, global_step)
                global_step += 1
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


def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return

    saved_model_weights = torch.load('./weight/adalora/resnet50/base/adalora_resnet50_base_8.pth', map_location='cpu')

    incremented_n_class = determine_n_class(saved_model_weights)
    config, model, optim, sche, model_loss, saver = load_framework(net_name, incremented_n_class)

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

    train(config, model, optim, sche, model_loss)
    exit()


if __name__ == '__main__':
    main()
