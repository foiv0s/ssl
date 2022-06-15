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
           train_loader, test_loader, stat_tracker, log_dir, device, nmb_crops, warmup, amp, lam):
    '''
    Training loop for optimizing encoder
    '''
    # If mixed precision is on, will add the necessary hooks into the model
    # and optimizer for half() conversions
    mix_precision = MixedPrecision(amp)
    mix = mix_precision.get_precision()

    # optimizer_disc = optim.Adam(model.discriminator.parameters(), betas=(0.8, 0.999), weight_decay=1e-5)
    # get target LR for LR warmup -- assume same LR for all param groups
    lr_real = [optimizer.param_groups[i]['lr'] for i in range(len(optimizer.param_groups))]
    torch.cuda.empty_cache()
    warmup_lam = 10
    warmup_lam_schedule = np.linspace(0., lam, len(train_loader) * warmup_lam)

    lam_schedule = np.concatenate((warmup_lam_schedule, np.ones(len(train_loader) * (epochs - warmup_lam))))
    dec_lam = 40
    lam_dec = np.linspace(lam, 1, len(train_loader) * dec_lam)

    # warmup_ = warmup * len(train_loader)
    # prepare checkpoint and stats accumulator
    # init_warm = np.linspace(1, lam, len(train_loader) * warmup)
    next_epoch = checkpointer.get_current_position()
    if next_epoch == 0:
        checkpointer.track_new_optimizer(optimizer)
        total_updates = 0
    else:
        optimizer = checkpointer.restore_optimizer_from_checkpoint(optimizer)
        total_updates = len(train_loader) * next_epoch
        # scheduler_inf.step(total_updates)
    stat_tracker.info(model.encoder)
    '''
    for _, batch in enumerate(train_loader):
        ((aug_imgs, imgs), labels, idxs) = batch
        aug_imgs = [aug_img.to(device) for aug_img in aug_imgs]
        model.init_membank(aug_imgs, idxs, train_loader.dataset.data.shape[0], nmb_crops=nmb_crops[0])
    model.generate_lbls()
    '''
    # model.init_membank(train_loader.dataset.data.shape[0])
    tracker = {'train_acc': [], 'test_acc': [], 'zeros': [], '5_nn': [],
               '10_nn': [], '50_nn': [], '100_nn': [], '200_nn': [], '500_nn': []}
    for epoch in range(next_epoch, epochs):
        epoch_stats = AverageMeterSet()
        epoch_updates = 0
        time_start = time.time()
        time_start1 = time.time()
        lbls, lbls_c = [], []
        model.step = epoch
        # stat_tracker.info(model._lam)
        # model.update_prototypes()
        for it, batch in enumerate(train_loader):
            ((aug_imgs, imgs), labels, idxs) = batch
            # get data and info about this minibatch
            lbls.append(labels.numpy())
            labels = torch.cat([labels] * (nmb_crops[0])).to(device)
            aug_imgs = [aug_img.to(device) for aug_img in aug_imgs]
            iteration = epoch * len(train_loader) + it
            for j, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = scheduler_inf[iteration][j]
            # run forward pass through model to get global and local features
            with mix:
                res_dict = model(x=aug_imgs, class_only=False, nmb_crops=nmb_crops, idxs=idxs)
                loss_cls = loss_xent(res_dict['class'], labels, clip=10)
                # res_dict['norm'] = torch.norm(res_dict['Z'],1,-1).mean()
                loss_opt = loss_cls + res_dict['loss'] * lam + res_dict['swav']#* lam_schedule[iteration]
                # lam_schedule[it] * lam_schedule[iteration]
                # + res_dict['norm'] * 0.01#1e-05
                # loss_opt = loss_cls + lam * res_dict['loss'] + res_dict['swav']
                # loss_opt = loss_cls + lam * res_dict['loss'] + res_dict['swav'] + res_dict['norm'] # * lam_schedule[iteration]
                # loss_opt = loss_cls + res_dict['loss'] + (1. / lam) * res_dict['swav']

            '''
            if epoch < warmup:
                lr_scale = min(1., float(total_updates + 1) / float(warmup_))
                for i, pg in enumerate(optimizer.param_groups):
                    pg['lr'] = lr_scale * lr_real[i]
            #'''

            optimizer.zero_grad()
            mix_precision.backward(loss_opt)
            # mix_precision.scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)

            # for p in model.encoder.project.parameters():
            #    torch.nn.utils.clip_grad_norm_(p, mix_precision.get_scale() * 0.2)
            #    torch.nn.utils.clip_grad_norm_(p, mix_precision.get_scale()*0.2)
            # for p in model.predictor.parameters():
            #    torch.nn.utils.clip_grad_norm_(p, mix_precision.get_scale() * 0.2)

            mix_precision.step(optimizer)
            # optimizer.step()
            # if amp:
            #    with apex.am.scale_loss(loss_opt, optimizer) as scaled_loss:
            #        scaled_loss.backward()
            #        # clc_loss.backward()
            # else:
            #    loss_opt.backward()

            '''
            model.zero_grad()
            res_dict['loss'].backward(retain_graph=True)
            for p in model.encoder.project.project.parameters():
                mmd_g = torch.norm(p.grad, 2, 1).sum()
                break
            model.zero_grad()
            res_dict['swav'].backward(retain_graph=True)
            for p in model.encoder.project.project.parameters():
                sim_g = torch.norm(p.grad, 2, 1).sum()
                break
            model.zero_grad()
            #'''
            # + res_dict['norm']  # + res_dict['prototypes']  #
            ## + res_dict['prototypes']  # + res_dict['adapt']
            '''
            if epoch < warmup:
                # for prototype in model.prototypes:
                # model.encoder_q.module.prototypes.prototypes.weight.grad = None
                pass
            else:
                # for prototype in model.prototypes:
                #    torch.nn.utils.clip_grad_norm_(prototype.prototypes.weight, 0.001)
                #    # stat_tracker.info(np.linalg.norm(model.prototypes.prototypes.weight.grad.detach().cpu().numpy()))
                pass
            '''
            # mix_precision.step(optimizer)

            # loss_opt.backward(retain_graph=True)
            # loss_opt.backward()

            # optimizer_disc.zero_grad()
            # loss_disc.backward()
            # optimizer_disc.step()

            with torch.no_grad():
                p = ((res_dict['Z'].detach() == 0).sum() / float(np.product(res_dict['Z'].detach().shape)))

            res_dict['class'] = res_dict['class'][:imgs.size()[0]].argmax(-1).cpu().numpy()
            lbls_c.append(res_dict['class'])
            # record loss and accuracy on minibatch
            epoch_stats.update_dict({'loss_cls': loss_cls.item(),
                                     'swav': res_dict['swav'].item(),
                                     'loss': res_dict['loss'].item(),
                                     'norm': res_dict['norm'].item(),
                                     # 'pr_std': model.encoder.project.get_weight().std().item(),
                                     # 'pr_max': torch.abs(model.encoder.project.get_weight()).max().item(),
                                     # 'grd_p': torch.norm(model.encoder.project.get_weight().grad).item(),
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
                # stat_tracker.info(fast_stats.averages(total_updates, prefix='fast/'))
                eval_time = time.time() - eval_start
                stat_str = fast_stats.pretty_string()
                stat_str = '-- {0:d} updates, eval_time {1:.2f}: {2:s}'.format(
                    total_updates, eval_time, stat_str)
                stat_tracker.info(stat_str)
        time_stop = time.time()
        spu = (time_stop - time_start1) / 100.
        stat_tracker.info('Epoch {0:d}, {1:.4f} min/epoch'.format(epoch, spu))
        # update learning rate
        # if epoch >= warmup:
        # scheduler_inf.step()
        lbls, lbls_c = np.concatenate(lbls).ravel(), np.concatenate(lbls_c).ravel()
        tracker['test_acc'].append(test_model(model, test_loader, device, epoch_stats, max_evals=500000, warmup=False))
        tracker['train_acc'].append(update_train_accuracies(epoch_stats, lbls, lbls_c))
        if (epoch % 10) == 0:
            knn(model, train_loader, test_loader, tracker, stat_tracker=stat_tracker)
        epoch_str = epoch_stats.pretty_string()
        tracker['zeros'].append(epoch_stats.avgs['zeros'])
        filename = log_dir + '/' + model.get_details(lam) + '.pkl'
        outfile = open(filename, 'wb')
        pickle.dump(tracker, outfile)
        outfile.close()
        diag_str = 'Epoch {0:d}: {1:s}'.format(epoch, epoch_str)
        stat_tracker.info(diag_str)
        sys.stdout.flush()
        # stat_tracker.info(epoch_stats.averages(epoch, prefix='costs/'))
        checkpointer.update(epoch + 1)


def train_self_supervised(model, learning_rate, train_loader, nmb_crops, test_loader, stat_tracker, checkpointer,
                          log_dir, device, warmup, epochs, amp, lam, wd, larc_):
    mods_inf = [m for m in model.info_modules]
    mods_cls = [m for m in model.class_modules]
    # mods_to_opt = mods_inf + mods_cls
    # learning_rate = [learning_rate] * len(mods_to_opt)
    # learning_rate[-1] = 0.02

    # model.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.encoder)
    # knn(model, train_loader, test_loader,stat_tracker=stat_tracker)
    # '''
    from graphs import SpectralNorm
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
    for mods in model.info_modules:
        pro_wd += mods.parameters()

    '''
    for model_ in mods_inf:
        for m in model_.modules():
            if isinstance(m, torch.nn.Conv2d) and hasattr(m, 'weight') and m.weight.requires_grad:
                wd_params.append(m.weight)
            if isinstance(m, torch.nn.Conv2d) and m.bias is not None:
                no_wd.append(m.bias)
            if isinstance(m, torch.nn.Linear) and hasattr(m, 'weight') and m.weight.requires_grad:
                pro_wd.append(m.weight)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                pro_wd.append(m.bias)
            if isinstance(m, SpectralNorm) and m.module.weight_bar.requires_grad:
                pro_wd.append(m.module.weight_bar)
                pro_wd.append(m.module.bias)
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm, torch.nn.BatchNorm1d)) and m.weight is not None:
                no_wd.append(m.weight)
                no_wd.append(m.bias)

    '''
    # '''
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.4, momentum=0.9, weight_decay=1e-06)
    # optimizer = optim.SGD([{'params': list(wd_params), 'lr': learning_rate[0], 'weight_decay': 0.0001},
    #                             {'params': list(no_wd), 'lr': learning_rate[0], 'weight_decay': 0},
    #                             {'params': model.evaluator.parameters(), 'lr':learning_rate[-1], 'weight_decay': 0.}],
    #                              momentum = 0.9)
    # '''
    optimizer = optim.SGD([{'params': wd_params, 'lr': learning_rate[0], 'weight_decay': wd[0]},
                           {'params': no_wd, 'lr': learning_rate[1], 'weight_decay': wd[1], 'exclude_adapting': True},
                           {'params': pro_wd, 'lr': learning_rate[2], 'weight_decay': wd[2]},
                           # {'params': list(all_params), 'lr': learning_rate[0], 'weight_decay': wd},
                           {'params': model.evaluator.parameters(), 'lr': learning_rate[3], 'weight_decay': 0.,
                            'exclude_adapting': True}],
                          momentum=0.9)
    # '''
    if larc_:
        optimizer = LARS(optimizer, trust_coefficient=0.001, clip=False)

    '''
    optimizer = optim.Adam(
        [{'params': mod.parameters(), 'lr': learning_rate} for i, mod in enumerate(mods_to_opt)], betas=(0.8, 0.999),
        weight_decay=1e-5)
    scheduler = MultiStepLR(optimizer, milestones=[600, 750, 900], gamma=0.4)

    # '''
    # learning_rate = np.array([0.5, 0.5, 0.5, 0.1])
    # learning_rate = np.array([0.05, 0.05, 0.05, 0.01])
    # optimizer = optim.SGD(
    #    [{'params': mod.parameters(), 'lr': learning_rate} for i, mod in enumerate(mods_to_opt)],
    #    momentum=0.9, weight_decay=1e-5)
    # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

    # '''
    final_lr = learning_rate * 1e-05
    lr = learning_rate
    warmup_lr_schedule = np.linspace(1e-10, learning_rate, len(train_loader) * warmup)
    iters = np.arange(len(train_loader) * (epochs - warmup))
    cosine_lr_schedule = np.array([final_lr + 0.5 * (lr - final_lr)
                                   * (1 + np.cos(np.pi * t / (len(train_loader) * (epochs - warmup)))) for t in iters])
    scheduler = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    # '''
    # if amp:
    #    model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")

    # scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0)

    # train the model
    _train(model, optimizer, scheduler, checkpointer, epochs,
           train_loader, test_loader, stat_tracker, log_dir, device, nmb_crops, warmup, amp, lam)


'''
class CosineAnnealingLR_(CosineAnnealingLR):

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super(CosineAnnealingLR_, self).__init__(optimizer, T_max, eta_min=eta_min, last_epoch=last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch == 0:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [group['lr'] + (base_lr - eta_min) *
                    (1 - math.cos(math.pi / self.T_max)) / 2
                    for base_lr, group, eta_min in
                    zip(self.base_lrs, self.optimizer.param_groups, self.eta_min)]
        return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
                (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
                (group['lr'] - self.eta_min[i]) + self.eta_min[i]
                for i, group in enumerate(self.optimizer.param_groups)]

    def _get_closed_form_lr(self):
        return [self.eta_min[i] + (base_lr - self.eta_min[i]) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for i, base_lr in enumerate(self.base_lrs)]
'''
