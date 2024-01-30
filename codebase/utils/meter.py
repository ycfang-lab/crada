"""
@author: Jiahua Wu
@contact: jhwu@shu.edu.cn
"""

from typing import Optional
import tqdm

class AverageMeter(object):
    r"""Computes and stores the average and current value.

    Examples::

        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """
    def __init__(self, name: str, fmt: Optional[str] = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", cur_tqdm:tqdm.tqdm = None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.tqdm = cur_tqdm

    def display(self, batch):
        if self.tqdm is not None:
            self.tqdm.set_description(self.prefix)
            entries = [str(meter) for meter in self.meters]
            self.tqdm.set_postfix_str(', '.join(entries))
        else:
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
            entries += [str(meter) for meter in self.meters]
            print('\t'.join(entries))


    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'