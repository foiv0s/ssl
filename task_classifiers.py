import sys
import time
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
# from apex.parallel.LARC import LARC
import numpy as np
from utils import test_model, knn
from tools.stats import AverageMeterSet, update_train_accuracies
from tools.costs import loss_xent


def _train(model, optimizer, scheduler_inf, checkpointer, epochs,
           train_loader, test_loader, stat_tracker, device, nmb_crops):
    torch.cuda.empty_cache()

    for epoch in range(epochs):
        epoch_stats = AverageMeterSet()
        time_start = time.time()
        lbls, lbls_c = [], []
        model.step = epoch
        model.encoder.eval()
        for _, batch in enumerate(train_loader):
            ((aug_imgs, imgs), labels, idxs) = batch
            # get data and info about this minibatch
            lbls.append(labels.numpy())
            labels = torch.cat([labels] * (nmb_crops[0])).to(device)
            aug_imgs = torch.cat([aug_img.to(device) for aug_img in aug_imgs[:nmb_crops[0]]])

            res_dict = model(x=aug_imgs, class_only=True)
            res_dict['class'] = model.evaluator(res_dict['emb'])

            loss_cls = loss_xent(res_dict['class'], labels)
            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()
            res_dict['class'] = res_dict['class'][:imgs.size()[0]].argmax(-1).cpu().numpy()
            lbls_c.append(res_dict['class'])

            # record loss and accuracy on minibatch
            epoch_stats.update_dict({'loss_cls': loss_cls.item()}, n=1)

        time_stop = time.time()
        spu = (time_stop - time_start) / 100.
        stat_tracker.info('Epoch {0:d}, {1:.4f} min/epoch'.format(epoch, spu))
        lbls, lbls_c = np.concatenate(lbls).ravel(), np.concatenate(lbls_c).ravel()
        test_model(model, test_loader, device, epoch_stats, max_evals=500000, warmup=False)
        update_train_accuracies(epoch_stats, lbls, lbls_c)
        epoch_str = epoch_stats.pretty_string()
        diag_str = 'Epoch  {0:d}: {1:s}'.format(epoch, epoch_str)
        stat_tracker.info(diag_str)
        scheduler_inf.step()
        sys.stdout.flush()
        #stat_tracker.info(epoch_stats.averages(epoch, prefix='costs/'))
        #stat_tracker.info(epoch_stats.averages(epoch, prefix='costs/'))
        # checkpointer.update(epoch + 1)


def train_classifiers(model, learning_rate, train_loader, nmb_crops,
                      test_loader, stat_tracker, checkpointer, log_dir, device, warmup, epochs, amp, wd, larc_):
    from graphs import MLPClassifier
    knn(model, train_loader, test_loader, stat_tracker=stat_tracker)
    model.evaluator = MLPClassifier(model.hyperparams['n_classes'], model.encoder.emb_dim, p=0.)

    model.evaluator.to(device)
    learning_rate = learning_rate[4]
    # optimizer = optim.Adam(model.evaluator.parameters(), lr=0.0005, weight_decay=0)
    optimizer = optim.SGD(model.evaluator.parameters(), lr=learning_rate)
    # scheduler = MultiStepLR(optimizer, milestones=[80, 150], gamma=0.4)
    scheduler = CosineAnnealingLR(optimizer, epochs)
    model.encoder.eval()
    _train(model, optimizer, scheduler, checkpointer, epochs,
           train_loader, test_loader, stat_tracker, device, nmb_crops)
