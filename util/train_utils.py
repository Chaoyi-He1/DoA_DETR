import sys
from typing import Iterable

import torch
from torch.cuda import amp
import util.misc as utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, accumulate: int, max_norm: float = 0,
                    warmup=False, scaler=None):
    # model.to(device)
    model.train()
    criterion.train()

    for name, param in model.named_parameters():
        if param.is_sparse:
            print(f"{name} is a sparse tensor")

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        accumulate = 1

    nb = len(data_loader)  # number of batches

    metric_logger = utils.MetricLogger(delimiter=";  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for i, (samples, targets, path, shapes, img_index) in \
            enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        ni = i + nb * epoch  # number integrated batches (since train start)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # with amp.autocast(enabled=scaler is not None):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        if not torch.isfinite(losses_reduced_scaled):
            print("Loss is {}, stopping training".format(losses_reduced_scaled.item()))
            print(loss_dict_reduced)
            sys.exit(1)

        losses *= 1. / accumulate  # scale loss

        # backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
        else:
            losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # optimize
        # Update the weights every 64 images trained
        if ni % accumulate == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        metric_logger.update(loss=losses_reduced_scaled, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if ni % accumulate == 0 and lr_scheduler is not None:  # The first round uses the warmup training method
            lr_scheduler.step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
