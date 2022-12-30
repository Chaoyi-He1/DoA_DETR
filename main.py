import argparse
import datetime
import glob
import json
import math
import random
import time
from pathlib import Path

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from util.distributed_utils import torch_distributed_zero_first
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

import datasets
import util.misc as utils
from datasets import parse_data_cfg
from datasets.Dataset_cls import LoadImagesAndLabels
from datasets.coco_utils import get_coco_api_from_dataset
from models import build_model
from util.eval_utils import evaluate
from util.train_utils import train_one_epoch
from util.util import check_file


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lf', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')

    # Model parameters
    parser.add_argument('--weights', type=str, default='', help="initial weights path")
    # * Backbone
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--freeze-layers', type=bool, default=False,
                        help='Freeze non-output layers')
    parser.add_argument('--img-size', type=int, default=512, help='Image size')

    # * Transformer
    parser.add_argument('--hyp', type=str, default='cfg/cfg.yaml', help='hyperparameters path')

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
    parser.add_argument('--set_cost_direction', default=8, type=float,
                        help="direction coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--direction_loss_coef', default=8, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='data/my_data.data')
    parser.add_argument('--cache_images', default=True, help="cache images to RAM")
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)

    # distributed training parameters
    parser.add_argument('--savebest', type=bool, default=False, help='only save best checkpoint')
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    return parser


def model_info(model):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients

    try:  # FLOPS
        from thop import profile
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 512, 512),), verbose=False)
        fs = ', %.1f GFLOPS' % (macs / 1E9 * 2)
    except:
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


def main(args, hyp):
    utils.init_distributed_mode(args)
    if args.rank in [-1, 0]:
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=args.name)

    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    device = torch.device(args.device)
    if "cuda" not in device.type:
        raise EnvironmentError("not find GPU device for training.")

    wdir = "weights" + os.sep  # weights dir
    best = wdir + "best.pt"
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_dict = parse_data_cfg(args.dataset_file)
    train_path = data_dict["train"]
    test_path = data_dict["valid"]

    if args.rank in [-1, 0]:
        # Remove previous results
        for f in glob.glob(results_file) + glob.glob("tmp.pk"):
            os.remove(f)

    model, criterion, postprocessors = build_model(args, hyp)
    model.to(device)
    accumulate = max(round(64 / (args.world_size * args.batch_size)), 1)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    model_info(model)

    start_epoch = 0
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # If pretrained weights are specified, load pretrained weights
    if args.weights.endswith(".pt"):
        ckpt = torch.load(args.weights, map_location=device)
        # load model
        try:
            ckpt["model"] = {k: v for k, v in ckpt["model"].items()
                             if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (args.weights, args.hyp, args.weights)
            raise KeyError(s) from e
        if args.rank in [-1, 0]:
            # load results
            if ckpt.get("training_results") is not None:
                with open(results_file, "w") as file:
                    file.write(ckpt["training_results"])  # write results.txt
        # epochs
        start_epoch = ckpt["epoch"] + 1
        if args.epochs < start_epoch:
            print('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                  (args.weights, ckpt['epoch'], args.epochs))
            args.epochs += ckpt['epoch']  # finetune additional epochs
        if args.amp and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        del ckpt

    # Whether to freeze the weights and only train the weights of the Transformer
    if args.freeze_layers:
        for n, p in model.named_parameters():
            if "backbone" in n:
                p.requires_grad_(False)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # param_dicts = [
    #     {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    # ]
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(pg, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # Specify which epoch to start from

    # dataset
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    with torch_distributed_zero_first(opt.rank):
        dataset_train = LoadImagesAndLabels(path=train_path, 
                                            img_size=args.img_size, 
                                            batch_size=args.batch_size,
                                            augment=False,
                                            hyp=hyp,  # augmentation hyperparameters
                                            mosaic=False,
                                            cache_images=args.cache_images,
                                            rank=args.rank)
        # 验证集的图像尺寸指定为img_size(512)
        dataset_val = LoadImagesAndLabels(path=test_path, 
                                          img_size=args.img_size, 
                                          batch_size=args.batch_size,
                                          augment=False,
                                          hyp=hyp,  # augmentation hyperparameters
                                          mosaic=False,
                                          cache_images=args.cache_images,
                                          rank=args.rank)

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
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
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

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    # Check if files exist
    args.hyp = check_file(args.hyp)
    # args.dataset_file = check_file(args.dataset_file)

    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    print("freeze: ", args.weights)
    main(args, hyp)
