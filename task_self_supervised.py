import sys
import time
import torch
import torch.optim as optim
import numpy as np
from tools.LARC import LARS
from utils import test_model, knn
from tools.stats import AverageMeterSet, update_train_accuracies
from tools.costs import loss_xent
from tools.mixed_precision import MixedPrecision
import pickle


# import apex
# from apex.parallel.LARC import LARC


def _train(model, optimizer, scheduler_inf, checkpointer, epochs,
           train_loader, test_loader, stat_tracker, log_dir, device, nmb_crops, warmup, amp):
    '''
    Training loop for optimizing encoder
    '''
    mix_precision = MixedPrecision(amp)
    mix = mix_precision.get_precision()

    lr_real = [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))]
    torch.cuda.empty_cache()

    next_epoch = checkpointer.get_current_position()
    if next_epoch == 0:
        checkpointer.track_new_optimizer(optimizer)
        total_updates = 0
    else:
        optimizer = checkpointer.restore_optimizer_from_checkpoint(optimizer)
        total_updates = len(train_loader) * next_epoch
    stat_tracker.info(model.encoder)
    tracker = {'train_acc': [], 'test_acc': [], 'zeros': [], '5_nn': [],
               '10_nn': [], '50_nn': [], '100_nn': [], '200_nn': [], '500_nn': []}
    for epoch in range(next_epoch, epochs):
        epoch_stats = AverageMeterSet()
        epoch_updates = 0
        time_start = time.time()
        time_start1 = time.time()
        lbls, lbls_c = [], []
        model.step = epoch
        for it, batch in enumerate(train_loader):
            ((aug_imgs, imgs), labels, idxs) = batch
            lbls.append(labels.numpy())
            labels = torch.cat([labels] * (nmb_crops[0])).to(device)
            aug_imgs = [aug_img.to(device) for aug_img in aug_imgs]
            iteration = epoch * len(train_loader) + it
            for j, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = scheduler_inf[iteration][j]
            # run forward pass through model to get global and local features
            with mix:
                res_dict = model(x=aug_imgs, class_only=False, nmb_crops=nmb_crops)
                loss_cls = loss_xent(res_dict['class'], labels, clip=10)
                loss_opt = loss_cls + res_dict['loss']

            optimizer.zero_grad()
            mix_precision.backward(loss_opt)
            mix_precision.step(optimizer)

            with torch.no_grad():
                p = ((res_dict['Z'].detach() == 0).sum() / float(np.product(res_dict['Z'].detach().shape)))

            res_dict['class'] = res_dict['class'][:imgs.size()[0]].argmax(-1).cpu().numpy()
            lbls_c.append(res_dict['class'])
            # record loss and accuracy on minibatch
            epoch_stats.update_dict({'loss_cls': loss_cls.item(),
                                     'loss': res_dict['loss'].item(),
                                     'en_std': model.encoder.model.layer2[0].conv1.weight.std().item(),
                                     'en_max': torch.abs(model.encoder.model.layer2[0].conv1.weight).max().item(),
                                     'grd_2': torch.norm(model.encoder.model.layer2[0].conv1.weight.grad).item(),
                                     'grd_4': torch.norm(model.encoder.model.layer4[0].conv1.weight.grad).item(),
                                     'zeros': p.item(),
                                     'lr': optimizer.param_groups[0]['lr'],
                                     }, n=1)

            # shortcut diagnostics to deal with long epochs
            total_updates += 1
            epoch_updates += 1
            if (total_updates % 100) == 0:
                # IDK, maybe this helps?
                torch.cuda.empty_cache()
                time_stop = time.time()
                spu = (time_stop - time_start) / 100.
                stat_tracker.info('Epoch {0:d}, {1:d} updates -- {2:.4f} sec/update'
                                  .format(epoch, epoch_updates, spu))
                time_start = time.time()
            if (total_updates % 500) == 0:
                # record diagnostics
                eval_start = time.time()
                fast_stats = AverageMeterSet()
                test_model(model, test_loader, device, fast_stats, max_evals=100000,
                           warmup=False, stat_tracker=stat_tracker)
                eval_time = time.time() - eval_start
                stat_str = fast_stats.pretty_string()
                stat_str = '-- {0:d} updates, eval_time {1:.2f}: {2:s}'.format(
                    total_updates, eval_time, stat_str)
                stat_tracker.info(stat_str)
        time_stop = time.time()
        spu = (time_stop - time_start1) / 100.
        stat_tracker.info('Epoch {0:d}, {1:.4f} min/epoch'.format(epoch, spu))
        lbls, lbls_c = np.concatenate(lbls).ravel(), np.concatenate(lbls_c).ravel()
        tracker['test_acc'].append(test_model(model, test_loader, device, epoch_stats, max_evals=500000, warmup=False))
        tracker['train_acc'].append(update_train_accuracies(epoch_stats, lbls, lbls_c))
        if (epoch % 10) == 0:
            knn(model, train_loader, test_loader, tracker, stat_tracker=stat_tracker)
        epoch_str = epoch_stats.pretty_string()
        tracker['zeros'].append(epoch_stats.avgs['zeros'])
        diag_str = 'Epoch {0:d}: {1:s}'.format(epoch, epoch_str)
        stat_tracker.info(diag_str)
        sys.stdout.flush()
        checkpointer.update(epoch + 1)


def train_self_supervised(model, learning_rate, train_loader, nmb_crops, test_loader, stat_tracker, checkpointer,
                          log_dir, device, warmup, epochs, amp, wd, larc_):
    learning_rate = np.array(learning_rate)
    no_wd = list()
    wd_params = list()
    pro_wd = list()
    for m in model.encoder.model.modules():
        if isinstance(m, torch.nn.Conv2d) and hasattr(m, 'weight') and m.weight.requires_grad:
            wd_params.append(m.weight)
        if isinstance(m, torch.nn.Conv2d) and m.bias is not None:
            no_wd.append(m.bias)
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)) and m.weight is not None:
            no_wd.append(m.weight)
            no_wd.append(m.bias)
    no_wd += list(model.prototypes.parameters())
    for mods in model.mlp_modules:
        pro_wd += mods.parameters()

    optimizer = optim.SGD([{'params': wd_params, 'lr': learning_rate[0], 'weight_decay': wd[0]},
                           {'params': no_wd, 'lr': learning_rate[1], 'weight_decay': wd[1]},
                           {'params': pro_wd, 'lr': learning_rate[2], 'weight_decay': wd[2]},
                           {'params': model.evaluator.parameters(), 'lr': learning_rate[3], 'weight_decay': 0.}],
                          momentum=0.9)

    if larc_:
        optimizer = LARS(optimizer, trust_coefficient=0.001, clip=False)

    # Cosine lr
    final_lr = learning_rate * 1e-05
    lr = learning_rate
    warmup_lr_schedule = np.linspace(1e-10, learning_rate, len(train_loader) * warmup)
    iters = np.arange(len(train_loader) * (epochs - warmup))
    cosine_lr_schedule = np.array([final_lr + 0.5 * (lr - final_lr)
                                   * (1 + np.cos(np.pi * t / (len(train_loader) * (epochs - warmup)))) for t in iters])
    scheduler = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

    _train(model, optimizer, scheduler, checkpointer, epochs,
           train_loader, test_loader, stat_tracker, log_dir, device, nmb_crops, warmup, amp)
