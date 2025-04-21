from __future__ import print_function
import argparse
import os
import sys

import numpy as np
from scipy import misc


def cal_metric(known, novel, method=None):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = {}
    # FPR
    results['FPR'] = fpr_at_tpr95
    # AUROC
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results['AUROC'] = -np.trapz(1. - fpr, tpr)
    # DTERR
    results['DTERR'] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()
    # AUIN
    denom = tp + fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results['AUIN'] = -np.trapz(pin[pin_ind], tpr[pin_ind])
    # AUOUT
    denom2 = tp[0] - tp + fp[0] - fp
    denom2[denom2 == 0.] = -1.
    pout_ind = np.concatenate([[True], denom2 > 0., [True]])
    pout = np.concatenate([[0.], (fp[0] - fp)/denom2, [.5]])
    results['AUOUT'] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])
    return results


def get_curve(known, novel, method=None):
    known = np.sort(known)
    novel = np.sort(novel)
    num_k = known.shape[0]
    num_n = novel.shape[0]
    all_vals = np.concatenate((known, novel))
    all_vals.sort()

    tp = -np.ones(num_k + num_n + 1, dtype=int)
    fp = -np.ones(num_k + num_n + 1, dtype=int)
    tp[0], fp[0] = num_k, num_n

    k = n = 0
    for i in range(num_k + num_n):
        if k == num_k:
            tp[i+1:] = tp[i]
            fp[i+1:] = np.arange(fp[i]-1, -1, -1)
            break
        if n == num_n:
            tp[i+1:] = np.arange(tp[i]-1, -1, -1)
            fp[i+1:] = fp[i]
            break
        if novel[n] < known[k]:
            n += 1
            tp[i+1] = tp[i]
            fp[i+1] = fp[i] - 1
        else:
            k += 1
            tp[i+1] = tp[i] - 1
            fp[i+1] = fp[i]

    # enforce monotonicity on ties
    j = num_k + num_n - 1
    for _ in range(num_k + num_n - 1):
        if all_vals[j] == all_vals[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    # FPR at TPR95
    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]
    fpr95 = np.sum(novel > threshold) / float(num_n)
    return tp, fp, fpr95


def print_all_results(results, datasets, method):
    mtypes = ['FPR', 'AUROC', 'AUIN']
    avg = compute_average_results(results)
    print('OOD detection method:', method)
    print('           ', ' '.join(f'{m:>6s}' for m in mtypes))
    for ds, res in zip(datasets, results):
        print(f'{ds:12s}', ' '.join(f'{100*res[m]:6.2f}' for m in mtypes))
    print('AVG         ', ' '.join(f'{100*avg[m]:6.2f}' for m in mtypes))


def compute_average_results(all_results):
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']
    avg = {m: 0.0 for m in mtypes}
    for res in all_results:
        for m in mtypes:
            avg[m] += res[m]
    for m in mtypes:
        avg[m] /= len(all_results)
    return avg


def compute_traditional_ood(base_dir, in_dataset, out_datasets, method, name, args=None):
    in_path = os.path.join(base_dir, in_dataset, method, name, 'in_scores.txt')
    known = np.loadtxt(in_path)

    known_sorted = np.sort(known)
    num_k = known.shape[0]
    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_sorted[round(0.05 * num_k)]

    all_results = []
    for od in out_datasets:
        out_path = os.path.join(base_dir, in_dataset, method, name, od, 'out_scores.txt')
        novel = np.loadtxt(out_path)
        all_results.append(cal_metric(known, novel, method))

    print_all_results(all_results, out_datasets, method)


def compute_stat(base_dir, in_dataset, out_datasets, method, name):
    in_path = os.path.join(base_dir, in_dataset, method, name, 'in_scores.txt')
    known = np.loadtxt(in_path)
    print(f"ID mean: {known.mean():.4f}, std: {known.std():.4f}")

    means, stds = [], []
    for od in out_datasets:
        out_path = os.path.join(base_dir, in_dataset, method, name, 'nat', od, 'out_scores.txt')
        novel = np.loadtxt(out_path)
        means.append(novel.mean())
        stds.append(novel.std())
    print(f"OOD mean: {np.mean(means):.4f}, std: {np.mean(stds):.4f}")


def compute_in(base_dir, in_dataset, method, name):
    label_path = os.path.join(base_dir, in_dataset, method, name, 'in_labels.txt')
    try:
        labels = np.loadtxt(label_path)
    except Exception:
        print(f"[metrics] skipping compute_in: cannot load labels from {label_path}")
        return

    in_path = os.path.join(base_dir, in_dataset, method, name, 'in_scores.txt')
    known = np.loadtxt(in_path)
    known_sorted = np.sort(known)
    num_k = known.shape[0]
    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_sorted[round(0.05 * num_k)]

    nat_in_cond = (known > threshold).astype(np.float32)
    correct = (labels[:,0] == labels[:,1]).astype(np.float32)

    fnr    = np.sum(correct * (1.0 - nat_in_cond)) / max(np.sum(correct), 1)
    acc    = np.mean(correct)
    eteacc = np.sum(correct * nat_in_cond) / max(np.sum(nat_in_cond), 1)
    print(f'FNR: {fnr*100:6.2f}, Acc: {acc*100:6.2f}, End-to-end Acc: {eteacc*100:6.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pytorch Detecting Out-of-distribution examples'
    )
    parser.add_argument('--in-dataset', default="CIFAR-10", type=str)
    parser.add_argument('--name',       default="densenet",  type=str)
    parser.add_argument('--method',     default='energy',    type=str)
    parser.add_argument('--base-dir',   default='output/ood_scores', type=str)
    parser.add_argument('--epsilon',    default=8,            type=int)
    args = parser.parse_args()

    np.random.seed(1)
    out_datasets = ['SVHN', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd', 'places365']
    compute_traditional_ood(args.base_dir, args.in_dataset, out_datasets, args.method, args.name)
    compute_in(args.base_dir, args.in_dataset, args.method, args.name)
