# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .linear_head import LinearClsHead


@HEADS.register_module()
class MultiTaskHead(LinearClsHead):
    """Muti-task classifier head.

    Args:
        num_classes (List[int]): Number of categories excluding the background
            category, one for each task.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
        task_weights (List[float] | optional): If specified, determine the
            weights of each task's loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 task_weights=None,
                 *args,
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_tasks = len(num_classes)

        if task_weights is None:
            task_weights = [1.0] * self.num_tasks
        assert len(task_weights) == self.num_tasks
        self.task_weights = task_weights

        assert isinstance(num_classes, list) and \
            all(num_cls > 0 for num_cls in num_classes)

        # Initialize FC layers, one for each task
        self.fc_layers = []
        for i in range(self.num_tasks):
            fc = nn.Linear(self.in_channels, self.num_classes[i])
            setattr(self, f"fc{i}", fc)
            self.fc_layers.append(fc)

    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a list of tensors, each
                  with shape ``(num_samples, num_classes_i)``.
                - If post processing, the output is a List[List[Tensor]], where
                  the outer is example-level and the inner is task-level (in
                  the context of multi-task learning). Each innermost tensor
                  has shape ``(num_classes,)``.
        """
        x = self.pre_logits(x)

        preds = []
        for fc in self.fc_layers:
            pred = fc(x)

            if softmax and pred is not None:
                pred = F.softmax(pred, dim=1)
            if post_process:
                pred = self.post_process(pred)
            preds.append(pred)

        # Unfold
        if post_process:
            preds = list(zip(*preds))

        return preds

    def loss(self, cls_score, gt_label, **kwargs):
        assert gt_label.ndim == 2 and gt_label.shape[1] == len(cls_score) \
            == self.num_tasks
        num_samples = len(cls_score[0])
        losses = dict(loss=0)
        # compute loss
        for i in range(self.num_tasks):
            loss = self.compute_loss(
                cls_score[i], gt_label[:, i], avg_factor=num_samples, **kwargs)
            losses["loss"] += self.task_weights[i] * loss
            if self.cal_acc:
                # compute accuracy
                acc = self.compute_accuracy(cls_score[i], gt_label[:, i])
                assert len(acc) == len(self.topk)
                losses[f'accuracy_{i}'] = {
                    f'top-{k}': a
                    for k, a in zip(self.topk, acc)
                }
        return losses

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        assert gt_label.ndim == 2 and gt_label.shape[1] == self.num_tasks
        cls_score = [self.fc_layers[i](x) for i in range(self.num_tasks)]
        return self.loss(cls_score, gt_label, **kwargs)
