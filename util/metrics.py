from __future__ import print_function
import argparse
import os
import sys

from scipy import misc
import numpy as np


def cal_metric(known, novel, method=None):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()
    mtypes = ['FPR', 'AUROC', 'DTERR', 'AUIN', 'AUOUT']

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
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0] - fp)/denom, [.5]])
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

    # enforce monotonicity for equal values
    j = num_k + num_n - 1
    for _ in range(num_k + num_n - 1):
        if all_vals[j] == all_vals[j-1]:
            tp[j] = tp[j+1]
            fp[j] = fp[j+1]
        j -= 1

    # FPR at TPR 95%
    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]
    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def print_results(results, in_dataset, out_dataset, name, method):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    print('in_distribution: ' + in_dataset)
    print('out_distribution: ' + out_dataset)
    print('Model Name: ' + name)
    print('\n OOD detection method: ' + method)
    for m in mtypes:
        print(f' {m:6s}', end='')
    print()
    for m in mtypes:
        print(f' {100*results[m]:6.2f}', end='')
    print('\n')


def print_all_results(results, datasets, method):
    mtypes = ['FPR', 'AUROC', 'AUIN']
    avg = compute_average_results(results)
    print(' OOD detection method: ' + method)
    print('             ', end='')
    for m in mtypes:
        print(f' {m:6s}', end='')
    for r, ds in zip(results, datasets):
        print(f'\n{ds:12s}', end='')
        for m in mtypes:
            print(f' {100*r[m]:6.2f}', end='')
    print(f'\nAVG         ', end='')
    for m in mtypes:
        print(f' {100*avg[m]:6.2f}', end='')
    print('\n')


def compute_average_results(all_results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg = {m: 0.0 for m in mtypes}
    for res in all_results:
        for m in mtypes:
            avg[m] += res[m]
    for m in mtypes:
        avg[m] /= len(all_results)
    return avg


def compute_traditional_ood(base_dir, in_dataset, out_datasets, method, name, args=None):
    # load in-scores
    in_path = os.path.join(base_dir, in_dataset, method, name, 'in_scores.txt')
    known = np.loadtxt(in_path)

    known_sorted = np.sort(known)
    num_k = known.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_sorted[round(0.05 * num_k)]

    all_results = []
    for out_ds in out_datasets:
        out_path = os.path.join(base_dir, in_dataset, method, name, out_ds, 'out_scores.txt')
        novel = np.loadtxt(out_path)

        results = cal_metric(known, novel, method)
        all_results.append(results)

    print_all_results(all_results, out_datasets, method)


def compute_stat(base_dir, in_dataset, out_datasets, method, name):
    in_path = os.path.join(base_dir, in_dataset, method, name, 'in_scores.txt')
    known = np.loadtxt(in_path)
    print(f"ID mean: {known.mean():.4f}  std: {known.std():.4f}")

    ood_means = []
    ood_stds  = []
    for out_ds in out_datasets:
        out_path = os.path.join(base_dir, in_dataset, method, name, 'nat', out_ds, 'out_scores.txt')
        novel = np.loadtxt(out_path)
        ood_means.append(novel.mean())
        ood_stds.append(novel.std())

    print(f"OOD mean: {np.mean(ood_means):.4f}  std: {np.mean(ood_stds):.4f}")


def compute_in(base_dir, in_dataset, method, name):
    in_path   = os.path.join(base_dir, in_dataset, method, name, 'in_scores.txt')
    label_path = os.path.join(base_dir, in_dataset, method, name, 'in_labels.txt')

    known = np.loadtxt(in_path)
    labels = np.loadtxt(label_path)

    known_sorted = np.sort(known)
    num_k = known.shape[0]

    if method == 'rowl':
        threshold = -0.5
    else:
        threshold = known_sorted[round(0.05 * num_k)]

    nat_in_cond = (known > threshold).astype(np.float32)
    correct    = (labels[:,0] == labels[:,1]).astype(np.float32)

    fnr    = np.sum(correct * (1.0 - nat_in_cond)) / max(np.sum(correct), 1)
    acc    = np.mean(correct)
    eteacc = np.sum(correct * nat_in_cond) / max(np.sum(nat_in_cond), 1)

    print(f'FNR: {fnr*100:6.2f}, Acc: {acc*100:6.2f}, End-to-end Acc: {eteacc*100:6.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pytorch Detecting Out-of-distribution examples in neural networks'
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
