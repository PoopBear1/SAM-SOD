import numpy as np
import sys
import importlib
#from data_esod import ESOD_Test
import torch
import time
from progress.bar import Bar
import os
from collections import OrderedDict
import cv2
from PIL import Image

from base.framework_factory import load_framework
from base.data import Test_Dataset
from base.metric import *
from base.util import *
import matplotlib.pyplot as plt

def visualize_semantic_map(semantic_map):
    np.random.seed(42)  # 确保颜色映射的一致性
    # 为40个类别生成随机颜色映射
    colors = np.random.randint(0, 255, (41, 3), dtype=np.uint8)

    # 直接使用颜色映射数组，避免循环
    # 创建一个与semantic_map_np形状相匹配、但是每个像素是RGB颜色的新数组
    visualized_map_np = colors[semantic_map]
    return visualized_map_np
def test_model(model, test_sets, config, saver=None):
    model.eval()
    st = time.time()
    torch.manual_seed(42)

    for set_name, test_set in test_sets.items():
        save_folder = os.path.join(config['save_path'], set_name)
        check_path(save_folder)

        titer = test_set.size
        MR = CLS_MetricRecorder(n_classes=40)

        test_bar = Bar('Dataset {:10}:'.format(set_name), max=titer)
        for j in range(titer):
            image, gt, name = test_set.load_data(j)
            Y = model(image.cuda())
            preds = torch.argmax(Y['final'], dim=1).squeeze().cpu().numpy()

            MR.update(pred=preds, gt=gt)

            config['save'] = torch.rand(1) < 1
            # save predictions
            if config['save']:
                print("saving output")
                fnl_folder = os.path.join(save_folder, 'final')
                check_path(fnl_folder)
                im_path = os.path.join(fnl_folder, name + '.png')
                # 保存预测类别图像，可能需要将类别标签映射到颜色
                pred_img = visualize_semantic_map(preds)
                Image.fromarray(pred_img).save(im_path)

                if saver is not None:
                    saver(Y, gt, name, save_folder, config)
                    pass
            test_bar.next()

        acc, miou = MR.show(bit_num=3)
        print('  acc: {:.3f}, miou: {:.3f}'.format(acc, miou))

    print('Test using time: {}.'.format(round(time.time() - st, 3)))




# def test_model(model, test_sets, config, saver=None):
#     model.eval()
#     st = time.time()
#     # config['save'] = True
#     # print(config['save'])
#     # print(config['save_path'])
#     for set_name, test_set in test_sets.items():
#         save_folder = os.path.join(config['save_path'], set_name)
#         check_path(save_folder)
#
#         titer = test_set.size
#         MR = MetricRecorder(titer)
#         scores = []
#
#         test_bar = Bar('Dataset {:10}:'.format(set_name), max=titer)
#         for j in range(titer):
#             image, gt, name = test_set.load_data(j)
#             Y = model(image.cuda())
#             pred = Y['final'][0, 0].sigmoid_().cpu().data.numpy()
#
#             out_shape = gt.shape
#
#             #pred = np.array(Image.fromarray(pred).resize((out_shape[::-1]), resample=0))
#             pred = cv2.resize(pred, (out_shape[::-1]), interpolation=cv2.INTER_LINEAR)
#
#             pred, gt = normalize_pil(pred, gt)
#             pred = np.clip(np.round(pred * 255) / 255., 0, 1)
#             MR.update(pre=pred, gt=gt)
#
#             #scores.append(get_scores(pred, gt))
#             #print(get_scores(pred, gt))
#
#             # save predictions
#             if config['save']:
#                 fnl_folder = os.path.join(save_folder, 'final')
#                 check_path(fnl_folder)
#                 im_path = os.path.join(fnl_folder, name + '.png')
#                 Image.fromarray((pred * 255)).convert('L').save(im_path)
#
#                 if saver is not None:
#                     saver(Y, gt, name, save_folder, config)
#                     pass
#
#             Bar.suffix = '{}/{}'.format(j, titer)
#             test_bar.next()
#
#         #scores = np.array(scores)
#         #print(np.mean(scores, axis=0))
#
#         mae, (maxf, meanf, *_), sm, em, wfm = MR.show(bit_num=3)
#         # #print('  MAE: {}, Max-F: {}, Maen-F: {}, SM: {}, EM: {}, Fbw: {}.'.format(mae, maxf, meanf, sm, em, wfm))
#         print('  Max-F: {:.3f}, Maen-F: {:.3f}, Fbw: {:.3f}, MAE: {:.3f}, SM: {:.3f}, EM: {:.3f}.'.format(maxf, meanf, wfm, mae, sm, em))
#
#     print('Test using time: {}.'.format(round(time.time() - st, 3)))

def main():
    if len(sys.argv) > 1:
        net_name = sys.argv[1]
    else:
        print('Need model name!')
        return

    config, model, _, _, _, saver = load_framework(net_name)
    print(config)

    #model.load_state_dict(torch.load(config['weight'], map_location='cpu'))
    saved_model = torch.load(config['weight'], map_location='cpu')
    new_name = {}
    for k, v in saved_model.items():
        if k.startswith('model'):
            new_name[k[6:]] = v
        else:
            new_name[k] = v
    model.load_state_dict(new_name)

    test_sets = OrderedDict()
    for set_name in config['vals']:
        test_sets[set_name] = Test_Dataset(name=set_name, config=config)

    model = model.cuda()

    test_model(model, test_sets, config, saver=saver)

if __name__ == "__main__":
    main()