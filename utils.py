import torch
import numpy as np
from sklearn.cluster import KMeans
import warnings
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.decomposition import PCA

eps = 1e-17
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def test_model(model, test_loader, device, stats, max_evals=200000, warmup=True, stat_tracker=None):
    '''
    Evaluate accuracy on test set
    '''
    # warm up batchnorm stats based on current model
    if warmup:
        _warmup_batchnorm(model, test_loader, device, batches=50, train_loader=False)

    def get_correct_count(lgt_vals, lab_vals):
        # count how many predictions match the target labels
        max_lgt = torch.max(lgt_vals.cpu().data, 1)[1]
        num_correct = (max_lgt == lab_vals).sum().item()
        return num_correct

    # evaluate model on test_loader
    model.eval()
    correct_glb_lin = 0.
    total = 0.
    y_, y, z, h = [], [], [], []
    for _, (images, labels, _) in enumerate(test_loader):
        if total > max_evals:
            break
        images = images.to(device)
        labels = labels.cpu()
        eval_idxs = labels >= 0
        with torch.no_grad():
            res_dict = model(x=images, class_only=True)
        # check classification accuracy
        y_.append(res_dict['class'][eval_idxs].cpu().numpy())
        y.append(labels[eval_idxs].cpu().numpy())
        correct_glb_lin += get_correct_count(res_dict['class'], labels)
        total += labels.size(0)
        z.append(res_dict['emb'][[eval_idxs]].cpu().numpy())
        h.append(res_dict['emb_'][[eval_idxs]].cpu().numpy())
    acc_glb_lin = correct_glb_lin / total
    model.train()
    '''
    y, y_ = np.concatenate(y).ravel(), np.concatenate(y_).ravel()
    z = np.concatenate(z)
    z_ = PCA(n_components=20).fit_transform(z)
    kmeans = KMeans(n_clusters=y.max() + 1, n_init=10)
    lbls__ = kmeans.fit_predict(z_)  # [:10000]
    stats.update('kmeans_accuracy', acc(y, lbls__), n=1)
    
    # record stats in the provided stat tracker
    mu, std = [], []
    for i in range(y.max() + 1):
        idxs = np.where(y == i)
        mu.append(z[idxs].mean(0)), std.append(z[idxs].std(0).sum())
    mu = np.array(mu)
    std = np.array(std)
    aa = (mu ** 2).sum(-1, keepdims=True)
    dist = (aa + aa.T - 2 * np.matmul(mu, mu.T))
    stat_tracker.info('dist', dist.sum() / (dist.shape[0] * (dist.shape[0] - 1)), std.mean())
    z = torch.from_numpy(z).cuda()
    y = torch.from_numpy(y).cuda()
    aa = (z ** 2).sum(-1, keepdims=True)
    dist = - 2 * torch.matmul(z, z.T)
    dist += aa
    dist += aa.T
    aa = dist.argsort(-1)
    ss = {}
    for i in [5, 50, 100, 200, 500]:
        score = (y.reshape(-1, 1) == y[aa][:, 1:i + 1]).sum().item()
        ss['score_' + str(i)] = score / (i * y.shape[0])
    stat_tracker.info(ss)
    '''
    stats.update('test_accuracy_linear_classifier', acc_glb_lin, n=1)
    return acc_glb_lin


def _warmup_batchnorm(model, data_loader, device, batches=100, train_loader=False):
    '''
    Run some batches through all parts of the model to warmup the running
    stats for batchnorm layers.
    '''
    model.train()
    for i, (images, _, _) in enumerate(data_loader):
        if i == batches:
            break
        if train_loader:
            images = images[0]
        images = images.to(device)
        _ = model(x=images, class_only=True)


# '''
def sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        Q = shoot_infs_v2(Q)
        sum_Q = torch.sum(Q)
        Q /= sum_Q
        r = torch.ones(Q.shape[0]).cuda() / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda() / (-1 * Q.shape[1])
        for it in range(nmb_iters):
            u = torch.sum(Q, dim=1)
            u = r / u
            u = shoot_infs_v2(u)
            Q *= u.unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
        # return (Q / torch.sum(Q)).float()


def shoot_infs_v2(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    if mask_inf.sum() > 0.:
        inp_tensor[mask_inf] = 0
        m = torch.max(inp_tensor)
        inp_tensor[mask_inf] = m
    return inp_tensor


def acc(y_true, y_pred, detailed=False):
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    if detailed:
        return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size, w, ind
    else:
        return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size


'''
Efficient implementation of knn through pytorch 
'''


@torch.no_grad()
def knn(model_, train_loader_, test_loader_, tracker=None, stat_tracker=None):
    if hasattr(train_loader_.dataset, 'targets'):
        n, m = len(train_loader_.dataset.targets), len(test_loader_.dataset.targets)
        transform = test_loader_.dataset.transform
        encoder, stack = model_.encoder, torch.stack
        y = torch.from_numpy(np.array(train_loader_.dataset.targets))
        y_ = torch.from_numpy(np.array(test_loader_.dataset.targets))
        z_train, z_test = [], []
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                for i in range(0, n, 1000):
                    imgs = [transform(img).type(torch.float16) for img in train_loader_.dataset.data[i:i + 1000]]
                    z_train.append(encoder(stack(imgs).cuda())[0])
                    del imgs
                for i in range(0, m, 1000):
                    imgs = [transform(img).type(torch.float16) for img in test_loader_.dataset.data[i:i + 1000]]
                    z_test.append(encoder(stack(imgs).cuda())[0])
                    del imgs
                torch.cuda.empty_cache()
            z_train = torch.cat(z_train)
            z_test = torch.cat(z_test)
            z_train_ = (z_train * z_train).sum(-1, keepdims=True)
            z_test_ = (z_test * z_test).sum(-1, keepdims=True)
            top_k = torch.zeros((m, 501), dtype=torch.int32)
            dist_k = torch.zeros((m, 501))
            for i in range(0, m, 1000):
                dist = - 2 * torch.matmul(z_test[i:i + 1000], z_train.T)
                dist += z_train_.T
                dist += z_test_[i:i + 1000]
                dist_k[i:i + 1000], top_k[i:i + 1000] = dist.topk(501, largest=False)
                # top_k[i:i + 1000] = dist.topk(501, largest=False)[1]
                del dist
                torch.cuda.empty_cache()
            del z_train_, z_test_
            top_k = top_k.type(dtype=torch.int64)
            ss, ss1 = {}, {}
            eye = torch.eye(y.max() + 1).numpy()
            dist_k = torch.exp(-0.2 * ((dist_k - dist_k.min(-1, keepdims=True)[0]) / dist_k.std(-1, keepdims=True)))
            torch.cuda.empty_cache()
            for i in [5, 10, 50, 100, 200, 500]:
                # score = np.around((y_.reshape(-1, 1) == y[aa][:, :i]).type(torch.float32).mean().item(), 4)
                tmp_eye = eye[y[top_k[:, :i]]]
                tmp_dist = (tmp_eye * dist_k[:, :i].unsqueeze(-1).numpy())
                score = (y_.cpu().numpy() == tmp_eye.sum(1).argmax(-1)).sum() / m
                score_p = (y_.cpu().numpy() == tmp_dist.sum(1).argmax(-1)).sum() / m
                ss['knn_score_' + str(i)], ss1['knn_score_p_' + str(i)] = score, score_p
                torch.cuda.empty_cache()
                if tracker is not None:
                    tracker[str(i) + '_nn'].append(score)
            stat_tracker.info(ss)
            stat_tracker.info(ss1)
            # del aa
            del top_k, dist_k
            torch.cuda.empty_cache()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    import argparse

    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
