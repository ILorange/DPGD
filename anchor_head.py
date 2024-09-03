# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import copy
import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from torch import Tensor
from mmdet.core import (anchor_inside_flags, build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, unmap)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from mmcv.ops import batched_nms
@HEADS.register_module()
class AnchorHead(BaseDenseHead, BBoxTestMixin):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01)):
        super(AnchorHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg,
                       'sampler') and self.train_cfg.sampler.type.split(
                           '.')[-1] != 'PseudoSampler':
                self.sampling = True
                sampler_cfg = self.train_cfg.sampler
                # avoid BC-breaking
                if loss_cls['type'] in [
                        'FocalLoss', 'GHMC', 'QualityFocalLoss'
                ]:
                    warnings.warn(
                        'DeprecationWarning: Determining whether to sampling'
                        'by loss type is deprecated, please delete sampler in'
                        'your config when using `FocalLoss`, `GHMC`, '
                        '`QualityFocalLoss` or other FocalLoss variant.')
                    self.sampling = False
                    sampler_cfg = dict(type='PseudoSampler')
            else:
                self.sampling = False
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        self.prior_generator = build_prior_generator(anchor_generator)
        featmap_strides = self.prior_generator.strides
        # print(featmap_strides)
        # print(featmap_strides)
        # print(featmap_strides)
        # print(featmap_strides)
        # print(featmap_strides)

        # Usually the numbers of anchors for each level are the same
        # except SSD detectors. So it is an int in the most dense
        # heads but a list of int in SSDHead
        self.num_base_priors = self.prior_generator.num_base_priors[0]
        self._init_layers()

    @property
    def num_anchors(self):
        warnings.warn('DeprecationWarning: `num_anchors` is deprecated, '
                      'for consistency or also use '
                      '`num_base_priors` instead')
        return self.prior_generator.num_base_priors[0]

    @property
    def anchor_generator(self):
        warnings.warn('DeprecationWarning: anchor_generator is deprecated, '
                      'please use "prior_generator" instead')
        return self.prior_generator

    def _init_layers(self):
        """Initialize layers of the head."""
        self.conv_cls = nn.Conv2d(self.in_channels,
                                  self.num_base_priors * self.cls_out_channels,
                                  1)
        self.conv_reg = nn.Conv2d(self.in_channels, self.num_base_priors * 4,
                                  1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores and bbox prediction.

                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_base_priors * 4.
        """
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list


    def _map_roi_levels(self, rois, num_levels):
        scale = torch.sqrt(
            (rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1))
        target_lvls = torch.floor(torch.log2(scale / 56 + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls


    def get_roi_mask(self, cls_scores, img_metas, gt_bboxes, phi=0.5):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        from mmdet.core import bbox_overlaps
        with torch.no_grad():
            anchor_list, _ = self.get_anchors(featmap_sizes, img_metas)
            mask_batch = []
            for batch in range(len(gt_bboxes)):
                mask_level = []
                target_lvls = self._map_roi_levels(gt_bboxes[batch], len(anchor_list[batch]))
                for level in range(len(anchor_list[batch])):
                    gt_level = gt_bboxes[batch][target_lvls == level]
                    h, w = featmap_sizes[level][0], featmap_sizes[level][1]
                    mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
                    if gt_level.shape[0] > 0:
                        # anchor_list[batch][level].to(torch.device("cuda:5"))
                        # gt_level.to(torch.device("cuda:5"))
                        IoU_map = bbox_overlaps(anchor_list[batch][level], gt_level)

                        max_iou, _ = torch.max(IoU_map, dim=0)
                        IoU_map = IoU_map.view(h, w, self.num_anchors, -1)
                        for ins in range(gt_level.shape[0]):
                            max_iou_per_gt = max_iou[ins] * phi
                            mask_per_gt = torch.sum(IoU_map[:, :, :, ins] > max_iou_per_gt, dim=2)
                            mask_per_img += mask_per_gt
                        mask_per_img = (mask_per_img > 0).double()
                    mask_level.append(mask_per_img)
                mask_batch.append(mask_level)
            mask_batch_level = []
            for i in range(len(mask_batch[0])):
                tmp = []
                for batch in range(len(mask_batch)):
                    tmp.append(mask_batch[batch][i])
                mask_batch_level.append(torch.stack(tmp, dim=0))

        return mask_batch_level

    def get_gt_mask(self, cls_scores, img_metas, gt_bboxes):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        #featmap_strides = [4,8,16,32,64]
        featmap_strides = [8, 16, 32, 64, 128]
        # print(featmap_strides)
        # print(featmap_strides.size())
        imit_range = [0, 0, 0, 0, 0]
        with torch.no_grad():
            mask_batch = []

            for batch in range(len(gt_bboxes)):
                mask_level = []
                target_lvls = self._map_roi_levels(gt_bboxes[batch], len(featmap_sizes))
                for level in range(len(featmap_sizes)):
                    gt_level = gt_bboxes[batch][target_lvls == level]  # gt_bboxes: BatchsizexNpointx4coordinate
                    h, w = featmap_sizes[level][0], featmap_sizes[level][1]
                    mask_per_img = torch.zeros([h, w], dtype=torch.double).cuda()
                    for ins in range(gt_level.shape[0]):
                        gt_level_map = gt_level[ins] / featmap_strides[level]
                        lx = max(int(gt_level_map[0]) - imit_range[level], 0)
                        rx = min(int(gt_level_map[2]) + imit_range[level], w)
                        ly = max(int(gt_level_map[1]) - imit_range[level], 0)
                        ry = min(int(gt_level_map[3]) + imit_range[level], h)
                        if (lx == rx) or (ly == ry):
                            mask_per_img[ly, lx] += 1
                        else:
                            mask_per_img[ly:ry, lx:rx] += 1
                    mask_per_img = (mask_per_img > 0).double()
                    mask_level.append(mask_per_img)
                mask_batch.append(mask_level)

            mask_batch_level = []
            for level in range(len(mask_batch[0])):
                tmp = []
                for batch in range(len(mask_batch)):
                    tmp.append(mask_batch[batch][level])
                mask_batch_level.append(torch.stack(tmp, dim=0))

        return mask_batch_level


    def reg_distill(self, stu_reg, tea_reg, tea_cls, stu_cls, gt_truth,img_metas):
        def select_single_mlvl(mlvl_tensors, batch_id, detach=True):
            assert isinstance(mlvl_tensors, (list, tuple))
            num_levels = len(mlvl_tensors)

            if detach:
                mlvl_tensor_list = [
                    mlvl_tensors[i][batch_id].detach() for i in range(num_levels)
                ]
            else:
                mlvl_tensor_list = [
                    mlvl_tensors[i][batch_id] for i in range(num_levels)
                ]
            return mlvl_tensor_list

        def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
            """Calculate overlap between two set of bboxes.

            If ``is_aligned `` is ``False``, then calculate the overlaps between each
            bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
            pair of bboxes1 and bboxes2.

            Args:
                bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
                bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
                    B indicates the batch dim, in shape (B1, B2, ..., Bn).
                    If ``is_aligned `` is ``True``, then m and n must be equal.
                mode (str): "iou" (intersection over union), "iof" (intersection over
                    foreground) or "giou" (generalized intersection over union).
                    Default "iou".
                is_aligned (bool, optional): If True, then m and n must be equal.
                    Default False.
                eps (float, optional): A value added to the denominator for numerical
                    stability. Default 1e-6.

            Returns:
                Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)

            Example:
                >>> bboxes1 = torch.FloatTensor([
                >>>     [0, 0, 10, 10],
                >>>     [10, 10, 20, 20],
                >>>     [32, 32, 38, 42],
                >>> ])
                >>> bboxes2 = torch.FloatTensor([
                >>>     [0, 0, 10, 20],
                >>>     [0, 10, 10, 19],
                >>>     [10, 10, 20, 20],
                >>> ])
                >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
                >>> assert overlaps.shape == (3, 3)
                >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
                >>> assert overlaps.shape == (3, )

            Example:
                >>> empty = torch.empty(0, 4)
                >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
                >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
                >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
                >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
            """

            assert mode in ['iou', 'iof', 'giou', 'diou'], f'Unsupported mode {mode}'
            # Either the boxes are empty or the length of boxes's last dimenstion is 4
            assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
            assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

            # Batch dim must be the same
            # Batch dim: (B1, B2, ... Bn)
            assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
            batch_shape = bboxes1.shape[:-2]

            rows = bboxes1.size(-2)
            cols = bboxes2.size(-2)
            if is_aligned:
                assert rows == cols

            if rows * cols == 0:
                if is_aligned:
                    return bboxes1.new(batch_shape + (rows,))
                else:
                    return bboxes1.new(batch_shape + (rows, cols))

            area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
                    bboxes1[..., 3] - bboxes1[..., 1])
            area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
                    bboxes2[..., 3] - bboxes2[..., 1])

            if is_aligned:
                lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
                rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

                wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
                overlap = wh[..., 0] * wh[..., 1]

                if mode in ['iou', 'giou']:
                    union = area1 + area2 - overlap
                else:
                    union = area1
                if mode == 'giou':
                    enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
                    enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
                if mode == 'diou':
                    enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
                    enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
                    b1_x1, b1_y1 = bboxes1[..., 0], bboxes1[..., 1]
                    b1_x2, b1_y2 = bboxes1[..., 2], bboxes1[..., 3]
                    b2_x1, b2_y1 = bboxes2[..., 0], bboxes2[..., 1]
                    b2_x2, b2_y2 = bboxes2[..., 2], bboxes2[..., 3]
            else:
                lt = torch.max(bboxes1[..., :, None, :2],
                               bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
                rb = torch.min(bboxes1[..., :, None, 2:],
                               bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

                wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
                overlap = wh[..., 0] * wh[..., 1]

                if mode in ['iou', 'giou']:
                    union = area1[..., None] + area2[..., None, :] - overlap
                else:
                    union = area1[..., None]
                if mode == 'giou':
                    enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                            bboxes2[..., None, :, :2])
                    enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                            bboxes2[..., None, :, 2:])
                if mode == 'diou':
                    enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                            bboxes2[..., None, :, :2])
                    enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                            bboxes2[..., None, :, 2:])
                    b1_x1, b1_y1 = bboxes1[..., :, None, 0], bboxes1[..., :, None, 1]
                    b1_x2, b1_y2 = bboxes1[..., :, None, 2], bboxes1[..., :, None, 3]
                    b2_x1, b2_y1 = bboxes2[..., None, :, 0], bboxes2[..., None, :, 1]
                    b2_x2, b2_y2 = bboxes2[..., None, :, 2], bboxes2[..., None, :, 3]

            eps = union.new_tensor([eps])
            union = torch.max(union, eps)

            ious = overlap / union

            if mode in ['iou', 'iof']:
                return ious
            # calculate gious
            if mode in ['giou']:
                enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
                enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
                enclose_area = torch.max(enclose_area, eps)
                gious = ious - (enclose_area - union) / enclose_area
                return gious
            if mode in ['diou']:
                left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
                right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
                rho2 = left + right
                enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
                enclose_c = enclose_wh[..., 0] ** 2 + enclose_wh[..., 1] ** 2
                enclose_c = torch.max(enclose_c, eps)
                dious = ious - rho2 / enclose_c
                return dious

        def giou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
            r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
            Box Regression <https://arxiv.org/abs/1902.09630>`_.

            Args:
                pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                    shape (n, 4).
                target (Tensor): Corresponding gt bboxes, shape (n, 4).
                eps (float): Epsilon to avoid log(0).

            Return:
                Tensor: Loss tensor.
            """
            gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
            loss = 1 - gious
            return loss

        featmap_sizes = [featmap.size()[-2:] for featmap in tea_cls]
        anchor_list, _ = self.get_anchors(featmap_sizes, img_metas)
        reg_total_loss = torch.tensor(0.).cuda()

        for i in range(len(gt_truth)):
            stu_bbox_batch = select_single_mlvl(stu_reg, i)
            tea_bbox_batch = select_single_mlvl(tea_reg, i)
            tea_cls_batch = select_single_mlvl(tea_cls, i)
            stu_cls_batch = select_single_mlvl(stu_cls, i)

            stu_rpn_bbox_batch = []
            tea_rpn_bbox_batch = []
            for _i in range(len(stu_bbox_batch)):
                stu_rpn_bbox_batch.append(stu_bbox_batch[_i].permute(1, 2, 0).reshape(-1, 4))
                tea_rpn_bbox_batch.append(tea_bbox_batch[_i].permute(1, 2, 0).reshape(-1, 4))
                tea_cls_batch[_i] = tea_cls_batch[_i].reshape(-1).sigmoid()
                stu_cls_batch[_i] = stu_cls_batch[_i].reshape(-1).sigmoid()

            stu_rpn_bbox_pred = torch.cat(stu_rpn_bbox_batch)
            tea_rpn_bbox_pred = torch.cat(tea_rpn_bbox_batch).detach()
            tea_rpn_cls_pred = torch.cat(tea_cls_batch).detach()
            stu_rpn_cls_pred = torch.cat(stu_cls_batch)
            anchor_rpn_list = torch.cat(anchor_list[i]).detach()

            stu_rpn_bbox_pred_decode = self.bbox_coder.decode(anchor_rpn_list, stu_rpn_bbox_pred,
                                                              max_shape=(600, 600, 3))
            tea_rpn_bbox_pred_decode = self.bbox_coder.decode(anchor_rpn_list, tea_rpn_bbox_pred,
                                                              max_shape=(600, 600, 3))

            stu_batch_iou = bbox_overlaps(stu_rpn_bbox_pred_decode, gt_truth[i], mode="iou")
            tea_batch_iou = bbox_overlaps(tea_rpn_bbox_pred_decode, gt_truth[i], mode="iou")

            if stu_batch_iou.shape[0] == 0 or stu_batch_iou.shape[1] == 0 or tea_batch_iou.shape[0] == 0 or tea_batch_iou.shape[0] == 0:
                continue
            # else:
            #     # max_iou, _ = torch.max(stu_batch_iou, dim=0)
            #     mask_per_img = torch.zeros([tea_rpn_bbox_pred_decode.shape[0], 1], dtype=torch.double).cuda()
            #     for ins in range(gt_truth[i].shape[0]):
            #         # max_iou_per_gt = max_iou[ins] * 0.6
            #         mask_per_gt = tea_batch_iou[:, ins] > 0.7
            #         mask_per_gt = mask_per_gt.float().unsqueeze(-1)
            #         mask_per_img += mask_per_gt


            # if tea_batch_iou.shape[0] == 0 or tea_batch_iou.shape[1] == 0:
            #     reg_loss = 0.0
            else:
                # max_iou, _ = torch.max(stu_batch_iou, dim=0)
                mask_per_img = torch.zeros([stu_rpn_bbox_pred_decode.shape[0], 1], dtype=torch.double).cuda()
                for ins in range(gt_truth[i].shape[0]):
                    # max_iou_per_gt = max_iou[ins] * 0.7
                    # mask_per_gt = stu_batch_iou[:, ins] > max_iou_per_gt
                    mask_per_gt = stu_batch_iou[:, ins] > 0.7
                    # is_all_false = mask_per_gt.all() == False

                    # if mask_per_gt.all() == False:
                    #     _, max_index = torch.max(stu_batch_iou[:, ins], dim=0)
                    #     mask_per_gt[max_index] = True

                    mask_per_gt = mask_per_gt.float().unsqueeze(-1)
                    mask_per_img += mask_per_gt

                # print(mask_per_img.size())
                mask_per_img = mask_per_img.squeeze(-1).nonzero().squeeze(-1)
                if mask_per_img.numel() == 0:
                    continue

                else:
                    stu_rpn_bbox_pred_decode = stu_rpn_bbox_pred_decode[mask_per_img]
                    tea_rpn_bbox_pred_decode = tea_rpn_bbox_pred_decode[mask_per_img]
                    tea_rpn_cls_pred = tea_rpn_cls_pred[mask_per_img]
                    stu_rpn_cls_pred = stu_rpn_cls_pred[mask_per_img]

                    # stu_batch_iou, _ = torch.max(stu_batch_iou, dim=1)
                    # tea_batch_iou, _ = torch.max(tea_batch_iou, dim=1)

                    # stu_batch_iou = stu_batch_iou[mask_per_img]
                    # tea_batch_iou = tea_batch_iou[mask_per_img]

                    reg_loss = giou_loss(stu_rpn_bbox_pred_decode, tea_rpn_bbox_pred_decode)
                    if reg_loss.numel() == 0:
                        continue
                    else:
                        weight = torch.abs(tea_rpn_cls_pred - stu_rpn_cls_pred)
                        # reg_loss = (reg_loss * weight).sum() / weight.sum()

                        reg_loss = reg_loss * weight
                        reg_loss = reg_loss.mean()
                        reg_total_loss += reg_loss


                    # mask = tea_batch_iou > stu_batch_iou
                    #
                    # if mask.numel() == 0:
                    #     continue
                    # else:
                    #     mask_stu_rpn_bbox_pred_decode = stu_rpn_bbox_pred_decode[mask]
                    #     mask_tea_rpn_bbox_pred_decode = tea_rpn_bbox_pred_decode[mask]
                    #     mask_tea_rpn_cls_pred = tea_rpn_cls_pred[mask]
                    #     mask_stu_rpn_cls_pred = stu_rpn_cls_pred[mask]
                    #
                    #     reg_loss = giou_loss(mask_stu_rpn_bbox_pred_decode, mask_tea_rpn_bbox_pred_decode)
                    #     if reg_loss.numel() == 0:
                    #         continue
                    #     else:
                    #
                    #         #weight = torch.abs(mask_tea_rpn_cls_pred - mask_stu_rpn_cls_pred)
                    #         # eps = 1e-7
                    #         # weight = torch.where(weight < eps, torch.tensor(eps).to(weight.device), weight)
                    #         # reg_loss = reg_loss * weight
                    #         reg_loss = reg_loss * torch.abs(mask_tea_rpn_cls_pred - mask_stu_rpn_cls_pred)
                    #
                    #         # reg_loss = torch.where(torch.isnan(reg_loss), torch.full_like(reg_loss, 0), reg_loss)
                    #         reg_loss = reg_loss.mean()
                    #         # eps = 1e-7
                    #         # reg_loss = torch.where(reg_loss < eps, torch.tensor(eps).to(reg_loss.device), reg_loss)
                    #         # print(reg_loss)
                    #         # eps = 1e-7
                    #         # reg_loss = torch.where(reg_loss < eps, torch.tensor(eps).to(reg_loss.device), reg_loss)
                    #         reg_total_loss += reg_loss


        return reg_total_loss

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'tea_cls_scores', 'tea_bbox_preds'))
    def get_bboxes_KD(self,
                      cls_scores,
                      bbox_preds,
                      tea_cls_scores,
                      tea_bbox_preds,
                      score_factors=None,
                      img_metas=None,
                      cfg=None,
                      rescale=False,
                      with_nms=True,
                      **kwargs):
        """Transform network outputs of a batch into bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Default None.
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        # list 5 (256,256) (128,128) (64,64) (32,23) (16,16)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            tea_cls_score_list = select_single_mlvl(tea_cls_scores, img_id)
            tea_bbox_pred_list = select_single_mlvl(tea_bbox_preds, img_id)

            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single_KD(cls_score_list, bbox_pred_list, tea_cls_score_list, tea_bbox_pred_list,
                                                 score_factor_list, mlvl_priors,
                                                 img_meta, cfg, rescale, with_nms,
                                                 **kwargs)
            result_list.append(results)
        return result_list

    def _get_bboxes_single_KD(self,
                              cls_score_list,
                              bbox_pred_list,
                              tea_cls_score_list,
                              tea_bbox_pred_list,
                              score_factor_list,
                              mlvl_anchors,
                              img_meta,
                              cfg,
                              rescale=False,
                              with_nms=True,
                              **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_anchors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has
                shape (num_anchors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RPN head does not need this value.
            mlvl_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_anchors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']

        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        nms_pre = cfg.get('nms_pre', -1)
        for level_idx in range(len(cls_score_list)):
            rpn_cls_score = cls_score_list[level_idx]
            rpn_bbox_pred = bbox_pred_list[level_idx]
            tea_rpn_cls_score = tea_cls_score_list[level_idx]
            tea_rpn_bbox_pred = tea_bbox_pred_list[level_idx]

            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            tea_rpn_cls_score = tea_rpn_cls_score.permute(1, 2, 0)

            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()

                tea_rpn_cls_score = tea_rpn_cls_score.reshape(-1)
                tea_scores = tea_rpn_cls_score.sigmoid()

            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)

                tea_rpn_cls_score = tea_rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
                tea_scores = tea_rpn_cls_score.softmax(dim=1)[:, 0]

            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            tea_rpn_bbox_pred = tea_rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            #mask = tea_scores > scores
            #indices = torch.nonzero(mask.squeeze(-1), as_tuple=False).squeeze(-1)
            # rpn_bbox_pred[indices] = tea_rpn_bbox_pred[mask.squeeze(-1)]
            # stu_tea_scores = torch.abs(scores - tea_scores)
            stu_tea_scores = scores
            anchors = mlvl_anchors[level_idx]
            if 0 < nms_pre < stu_tea_scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = stu_tea_scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                stu_tea_scores = ranked_scores[:nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(stu_tea_scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                stu_tea_scores.new_full((stu_tea_scores.size(0),),
                                        level_idx,
                                        dtype=torch.long))

        return self._bbox_post_process_KD(mlvl_scores, mlvl_bbox_preds,
                                          mlvl_valid_anchors, level_ids, cfg,
                                          img_shape)

    def _bbox_post_process_KD(self, mlvl_scores, mlvl_bboxes, mlvl_valid_anchors,
                              level_ids, cfg, img_shape, **kwargs):
        """bbox post-processing method.

        Do the nms operation for bboxes in same level.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            mlvl_valid_anchors (list[Tensor]): Anchors of all scale level
                each item has shape (num_bboxes, 4).
            level_ids (list[Tensor]): Indexes from all scale levels of a
                single image, each item has shape (num_bboxes, ).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, `self.test_cfg` would be used.
            img_shape (tuple(int)): The shape of model's input image.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bboxes)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)
        min_bbox_size = 0
        if min_bbox_size >= 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_mask = (w > min_bbox_size) & (h > min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() > 0:
            dets, _ = batched_nms(proposals, scores, ids, cfg.nms)
        else:
            return proposals.new_zeros(0, 5)

        return dets[:cfg.max_per_img]

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                # labels[pos_inds] = gt_labels[
                #     sampling_result.pos_assigned_gt_inds]
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all
                  images.
                - num_total_neg (int): Number of negative samples in all
                  images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5), where
                5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,), The length of list should always be 1.
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
