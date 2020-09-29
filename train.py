# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import logging
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import Dataset

import datasets
import util.misc as utils
from datasets import get_coco_api_from_dataset
from datasets.coco import make_coco_transforms
from engine import evaluate, train_one_epoch
from models import build_model

logger = logging.getLogger(__name__)


def parse_textds_json(json_path, image_root, profile=False):
    content = load_json(json_path)
    d = []
    for gt in content['data_list']:
        img_path = os.path.join(image_root, gt['img_name'])
        if not os.path.exists(img_path):
            if profile:
                logger.warning('image not exists - {} '.format(img_path))
            continue
        polygons = []
        texts = []
        legibility_list = []  # 清晰文字，默认：true
        language_list = []
        for annotation in gt['annotations']:
            if len(annotation['polygon']) == 0:  # or len(annotation['text']) == 0:
                continue
            polygons.append(np.array(annotation['polygon']).reshape(-1, 2))
            texts.append(annotation['text'])
            legibility_list.append(not annotation['illegibility'])
            language_list.append(annotation['language'])
            for char_annotation in annotation['chars']:
                if len(char_annotation['polygon']) == 0 or len(char_annotation['char']) == 0:
                    continue
                polygons.append(char_annotation['polygon'])
                texts.append(char_annotation['char'])
                legibility_list.append(not char_annotation['illegibility'])
                language_list.append(char_annotation['language'])
        if len(polygons) > 0:
            d.append({'img_path': img_path, 'polygons': np.array(polygons), 'texts': texts,
                      'legibility': legibility_list,
                      'language': language_list})
    return d


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = json.load(f)
    return content


class JsonDataset(Dataset):
    """

    """

    @classmethod
    def load_json(cls, json_path, mode='train'):
        d = []
        if isinstance(json_path, list):
            img_root = [os.path.join(os.path.dirname(p), mode) for p in json_path]
            for j_path, i_root in zip(json_path, img_root):
                d.extend(parse_textds_json(json_path=j_path, image_root=i_root))
        else:
            image_root = os.path.join(os.path.dirname(json_path), mode)
            d = parse_textds_json(json_path=json_path, image_root=image_root)
        return d

    def __init__(self, json_path, mode='train', language='english',
                 rect=False, psize=(640, 640), area_limit=80.0, scale=0.125,
                 auto_fit=False, dict_out=True, profile=False, transforms=None):
        super().__init__()
        self.mode = mode
        self.arbitary_text_mode = not rect

        self.out_img_size = psize
        self.auto_fit = auto_fit
        self.dict_item = dict_out
        self.min_area = area_limit
        self.scale = scale
        self.profile = profile

        self.data = self.load_json(json_path=json_path, mode=mode)

        self._transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['img_path']

        if not os.path.exists(image_path):
            if self.profile:
                logger.warning(f'NO Exists - {image_path}')
            return self.__getitem__(random.randint(0, len(self) - 1))

        text_polys = item['polygons']
        legibles = item['legibility']

        cvimg = cv2.imread(image_path)
        cvimg = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)

        if self._transforms is not None:
            cvimg, text_polys, legibles = self._transforms(cvimg, text_polys, legibles)

        return cvimg, text_polys, legibles


class SimpleJsonDataset(JsonDataset):
    def __init__(self, json_path, mode='train', language='english', rect=False, psize=(640, 640), area_limit=80.0,
                 scale=0.125, auto_fit=False, dict_out=True, profile=False, transforms=None, target_transforms=None):
        super().__init__(json_path, mode, language, rect, psize, area_limit, scale, auto_fit, dict_out, profile,
                         transforms)
        self._target_transforms = target_transforms

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['img_path']
        target = {'image_id': torch.tensor(idx, dtype=torch.int64)}

        if not os.path.exists(image_path):
            if self.profile:
                logger.warning(f'NO Exists - {image_path}')
            return self.__getitem__(random.randint(0, len(self) - 1))

        pimg = Image.open(image_path)
        cvimg = pimg.copy()
        pimg.close()
        target['orig_size'] = torch.tensor(pimg.size, dtype=torch.int64)

        classes = [0 for _ in item['polygons']]
        iscrowd = [0 for _ in item['polygons']]
        target['labels'] = torch.tensor(classes, dtype=torch.int64)
        target['iscrowd'] = torch.tensor(iscrowd)

        if 'bbox' not in item and 'polygons' in item:
            polys = item['polygons']
            if not isinstance(polys, np.ndarray):
                polys = np.array(polys)

            bboxes = [None] * polys.shape[0]
            for pi, poly in enumerate(polys):
                x0 = np.min(poly[..., 0])
                y0 = np.min(poly[..., 1])
                x1 = np.max(poly[..., 0])
                y1 = np.max(poly[..., 1])
                bboxes[pi] = (x0, y0, x1, y1)
            target['boxes'] = torch.as_tensor(bboxes, dtype=torch.float32).reshape(-1, 4)

        if self._transforms:
            cvimg, item = self._transforms(cvimg, target)

        return cvimg, item


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    train_json = '/opt/ocr-data/icpr-2018/train.json'
    val_json = '/opt/ocr-data/icpr-2018/val.json'
    dataset_train = SimpleJsonDataset(json_path=train_json, mode='train', transforms=make_coco_transforms('train'))
    dataset_val = SimpleJsonDataset(json_path=val_json, mode='val', transforms=make_coco_transforms('val'))

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    model.accumulate = 100
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        model.step = epoch * args.batch_size
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # test_stats, coco_evaluator = evaluate(
        #     model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        # )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     # **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            # if coco_evaluator is not None:
            #     (output_dir / 'eval').mkdir(exist_ok=True)
            #     if "bbox" in coco_evaluator.coco_eval:
            #         filenames = ['latest.pth']
            #         if epoch % 50 == 0:
            #             filenames.append(f'{epoch:03}.pth')
            #         for name in filenames:
            #             torch.save(coco_evaluator.coco_eval["bbox"].eval,
            #                        output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
