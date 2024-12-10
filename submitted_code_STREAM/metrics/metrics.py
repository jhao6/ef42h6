
from __future__ import print_function

import torch


def task_changes(result_t):
    n_tasks = int(result_t.max() + 1)
    changes = []
    current = result_t[0]
    for i, t in enumerate(result_t):
        if t != current:
            changes.append(i)
            current = t

    return n_tasks, changes


def confusion_matrix(avg_result_a,  result_a, fname=None):

    fgt = torch.zeros(result_a.size(0))
    for t in range(len(fgt)-1):
        fgt[t] = result_a[t, t] - result_a[result_a.size(0)-1, t]

    if fname is not None:
        f = open(fname, 'w')
        # print('Diagonal Accuracy: %.4f' % acc.mean(), file=f)
        print(f'Forgetting: {fgt}', file=f)
        print(f'Avg Acc: {avg_result_a}', file=f)
        f.close()

    stats = []

    stats.append(fgt.mean())

    return stats
