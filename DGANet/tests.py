from __future__ import print_function

import warnings

warnings.filterwarnings('ignore')
import argparse
import os
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
import time

import torch
from lib.config import config
from lib.config import update_config
from model.dganet import Base
from utils.loss import BCL

from utils import configs, dataset
from utils.util import AverageMeter, setSeed, get_confusion_matrix, intersectionAndUnionGPU


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config',
                        type=str,
                        default='./config/change/sysucd.yaml',
                        help='config file')
    parser.add_argument('opts',
                        help='',
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = configs.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = configs.merge_cfg_from_list(cfg, args.opts)
    update_config(config, args)
    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args
    torch.set_num_threads(1)
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    setSeed(args.manual_seed)

    model = Base(in_ch=3, fc_ch=64)
    print("Finish!\nParameter number is ", sum(param.numel() for param in model.parameters()))

    seg_loss = BCL().cuda()

    global logger, writer, best_accuracy, best_f1
    best_accuracy = 0
    best_f1 = 0
    logger = get_logger()
    writer = SummaryWriter(args.save_path)
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(2))

    if len(args.train_gpu) > 1:
        model = torch.nn.DataParallel(model.cuda())
    else:
        model = model.cuda()

    if args.weight:
        if os.path.isfile(args.weight):
            logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])

            logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    test_dataset = dataset.SemData(split='test',
                                   dataset=args.dataset,
                                   data_root=args.data_root,
                                   data_list=args.test_list,
                                   config=args,
                                   )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size_test,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              drop_last=False)

    test(test_dataset, test_loader, model, seg_loss, sv_pred=args.has_prediction)


def test(test_dataset, test_loader, model, seg_loss, sv_pred=False):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()

    intersection_meterc = AverageMeter()
    union_meterc = AverageMeter()
    target_meterc = AverageMeter()

    model.eval()

    end = time.time()
    confusion_matrixc = np.zeros((2, 2))
    test_loss = AverageMeter()
    with torch.no_grad():
        time_sum = 0
        for i, (image1, image2, labelchange, name) in enumerate(test_loader, 0):
            data_time.update(time.time() - end)
            image1 = image1.float().cuda()
            image2 = image2.float().cuda()
            labelchange = labelchange.float().cuda()
            time_start = time.time()
            outputschange, y = model(image1, image2)
            time_end = time.time()
            time_sum += time_end - time_start

            zero = torch.zeros_like(outputschange)
            one = torch.ones_like(outputschange)
            predictedc = torch.where(outputschange > args.threshold, one, zero)  #

            seg_gt = (labelchange > 0).float()
            loss = seg_loss(outputschange, seg_gt)

            n = args.batch_size_test
            test_loss.update(loss.item(), n)

            confusion_matrixc += get_confusion_matrix(
                labelchange,
                predictedc,
                labelchange.size(),
                2,
                255)

            intersectionc, unionc, targetc = intersectionAndUnionGPU(predictedc, labelchange.long().cuda(), 2,
                                                                     args.ignore_label)
            intersectionc, unionc, targetc = intersectionc.cpu().numpy(), unionc.cpu().numpy(), targetc.cpu().numpy()
            intersection_meterc.update(intersectionc), union_meterc.update(unionc), target_meterc.update(targetc)

            batch_time.update(time.time() - end)
            end = time.time()
            if sv_pred:
                save_path = args.save_folder
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                labchange_suffix = "_change.png"
                test_dataset.save_pred(args, np.squeeze(predictedc.cpu().numpy()),
                                       np.squeeze(labelchange.cpu().numpy()), save_path, name, suffix=labchange_suffix)

            if ((i + 1) % args.print_freq == 0) or i == len(test_loader) - 1:
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            .format(i + 1, len(test_loader),
                                    data_time=data_time,
                                    batch_time=batch_time))

        pos_change = confusion_matrixc.sum(1)  #
        res_change = confusion_matrixc.sum(0)  #
        tp_change = np.diag(confusion_matrixc)
        recall_change = (tp_change / pos_change)
        precision_change = (tp_change / res_change)
        f1_change = 2 * precision_change * recall_change / (precision_change + recall_change)

        allAcc = sum(intersection_meterc.sum) / (sum(target_meterc.sum) + np.finfo(np.float32).eps)
        IoUc = (intersection_meterc.sum / (union_meterc.sum + np.finfo(np.float32).eps))[1]

        print('change result: precision/recall/f1/allAcc/IoUc/Loss', precision_change[1], recall_change[1],
              f1_change[1], allAcc, IoUc, test_loss.avg)
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
