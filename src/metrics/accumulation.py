#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# @Project      : ObjectWatermark
# @File         : accumulation.py
# @Author       : chengxin
# @Email        : zcx_language@163.com
# @Reference    : None
# @CreateTime   : 2023/9/2 10:24
#
# Import lib here
from typing import Optional, Tuple
import torch
from torchmetrics import Metric


class Accumulation(Metric):
    def __init__(self):
        super().__init__()
        # Set to True if the metric is differentiable else set to False
        is_differentiable: Optional[bool] = None

        # Set to True if the metric reaches it optimal value when the metric is maximized.
        # Set to False if it when the metric is minimized.
        higher_is_better: Optional[bool] = True

        # Set to True if the metric during 'update' requires access to the global metric
        # state for its calculations. If not, setting this to False indicates that all
        # batch states are independent and we will optimize the runtime of 'forward'
        full_state_update: bool = True

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, n_correct: int, n_total: int):
        self.correct += n_correct
        self.total += n_total

    def compute(self):
        return self.correct.float() / self.total


def run():
    pass


if __name__ == '__main__':
    run()
