# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
import warnings
import torch
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.ops import batched_nms
from mmcv.runner import BaseModule, force_fp32
import copy
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl
from torch import Tensor

class BaseDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for DenseHeads."""

    def __init__(self, init_cfg=None):
        super(BaseDenseHead, self).__init__(init_cfg)

    def init_weights(self):
        super(BaseDenseHead, self).init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    @abstractmethod
    def loss(self, **kwargs):
        """Compute losses of the head."""
        pass

    def _map_roi_levels(self, rois, num_levels):
        scale = torch.sqrt(
            (rois[:, 2] - rois[:, 0] + 1) * (rois[:, 3] - rois[:, 1] + 1))
        target_lvls = torch.floor(torch.log2(scale / 56 + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def get_gt_mask(self, cls_scores, img_metas, gt_bboxes):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
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

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes_KD_single(self,
                             cls_scores,
                             bbox_preds,
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

            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single_KDFCOS(cls_score_list, bbox_pred_list,
                                                        score_factor_list, mlvl_priors,
                                                        img_meta, cfg, rescale, with_nms,
                                                        **kwargs)
         
            result_list.append(results)
        return result_list

    def _get_bboxes_single_KDFCOS(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process_KDFCOS(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process_KDFCOS(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)

                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            # print(det_bboxes.size())
            # print(det_labels.size())
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels



    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
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
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                              score_factor_list, mlvl_priors,
                                              img_meta, cfg, rescale, with_nms,
                                              **kwargs)
            result_list.append(results)
        return result_list

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, score_factor, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              score_factor_list, mlvl_priors)):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, mlvl_score_factors, **kwargs)

    def _bbox_post_process(self,
                           mlvl_scores,
                           mlvl_labels,
                           mlvl_bboxes,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           mlvl_score_factors=None,
                           **kwargs):
        """bbox post-processing method.

        The boxes would be rescaled to the original image scale and do
        the nms operation. Usually `with_nms` is False is used for aug test.

        Args:
            mlvl_scores (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_labels (list[Tensor]): Box class labels from all scale
                levels of a single image, each item has shape
                (num_bboxes, ).
            mlvl_bboxes (list[Tensor]): Decoded bboxes from all scale
                levels of a single image, each item has shape (num_bboxes, 4).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
            mlvl_score_factors (list[Tensor], optional): Score factor from
                all scale levels of a single image, each item has shape
                (num_bboxes, ). Default: None.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        assert len(mlvl_scores) == len(mlvl_bboxes) == len(mlvl_labels)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = torch.cat(mlvl_labels)

        if mlvl_score_factors is not None:
            # TODO： Add sqrt operation in order to be consistent with
            #  the paper.
            mlvl_score_factors = torch.cat(mlvl_score_factors)
            mlvl_scores = mlvl_scores * mlvl_score_factors

        if with_nms:
            if mlvl_bboxes.numel() == 0:
                det_bboxes = torch.cat([mlvl_bboxes, mlvl_scores[:, None]], -1)

                return det_bboxes, mlvl_labels

            det_bboxes, keep_idxs = batched_nms(mlvl_bboxes, mlvl_scores,
                                                mlvl_labels, cfg.nms)
            det_bboxes = det_bboxes[:cfg.max_per_img]
            det_labels = mlvl_labels[keep_idxs][:cfg.max_per_img]
            # print(det_bboxes.size())
            # print(det_labels.size())
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_labels

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)



    def reg_distill_single_retinanet(self, stu_reg, tea_reg, tea_cls, stu_cls, gt_truth,img_metas):
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
        # anchor_list, _ = self.get_anchors(featmap_sizes, img_metas)
        anchor_list = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=tea_cls[0].dtype,
            device=tea_cls[0].device)


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
                # cls_score = cls_score.permute(1, 2,
                #                               0).reshape(-1, self.cls_out_channels)
                tea_cls_batch[_i] = tea_cls_batch[_i].reshape(-1, self.cls_out_channels).sigmoid()
                stu_cls_batch[_i] = stu_cls_batch[_i].reshape(-1, self.cls_out_channels).sigmoid()

            stu_rpn_bbox_pred = torch.cat(stu_rpn_bbox_batch)
            tea_rpn_bbox_pred = torch.cat(tea_rpn_bbox_batch).detach()
            tea_rpn_cls_pred = torch.cat(tea_cls_batch).detach()
            stu_rpn_cls_pred = torch.cat(stu_cls_batch)
            anchor_rpn_list = torch.cat(anchor_list).detach()

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

    def reg_distill_single_fcos(self, stu_reg, tea_reg, tea_cls, stu_cls, gt_truth, img_metas):
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
        # anchor_list, _ = self.get_anchors(featmap_sizes, img_metas)
        anchor_list = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=tea_cls[0].dtype,
            device=tea_cls[0].device)

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
                # cls_score = cls_score.permute(1, 2,
                #                               0).reshape(-1, self.cls_out_channels)
                tea_cls_batch[_i] = tea_cls_batch[_i].reshape(-1, self.cls_out_channels).sigmoid()
                stu_cls_batch[_i] = stu_cls_batch[_i].reshape(-1, self.cls_out_channels).sigmoid()

            stu_rpn_bbox_pred = torch.cat(stu_rpn_bbox_batch)
            tea_rpn_bbox_pred = torch.cat(tea_rpn_bbox_batch).detach()
            tea_rpn_cls_pred = torch.cat(tea_cls_batch).detach()
            stu_rpn_cls_pred = torch.cat(stu_cls_batch)
            anchor_rpn_list = torch.cat(anchor_list).detach()

            stu_rpn_bbox_pred_decode = self.bbox_coder.decode(anchor_rpn_list, stu_rpn_bbox_pred,
                                                              max_shape=(600, 600, 3))
            tea_rpn_bbox_pred_decode = self.bbox_coder.decode(anchor_rpn_list, tea_rpn_bbox_pred,
                                                              max_shape=(600, 600, 3))

            stu_batch_iou = bbox_overlaps(stu_rpn_bbox_pred_decode, gt_truth[i], mode="iou")
            tea_batch_iou = bbox_overlaps(tea_rpn_bbox_pred_decode, gt_truth[i], mode="iou")

            if stu_batch_iou.shape[0] == 0 or stu_batch_iou.shape[1] == 0 or tea_batch_iou.shape[0] == 0 or \
                    tea_batch_iou.shape[0] == 0:
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

            # mask = tea_scores > scores
            # indices = torch.nonzero(mask.squeeze(-1), as_tuple=False).squeeze(-1)
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

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def onnx_export(self,
                    cls_scores,
                    bbox_preds,
                    score_factors=None,
                    img_metas=None,
                    with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            score_factors (list[Tensor]): score_factors for each s
                cale level with shape (N, num_points * 1, H, W).
                Default: None.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc. Default: None.
            with_nms (bool): Whether apply nms to the bboxes. Default: True.

        Returns:
            tuple[Tensor, Tensor] | list[tuple]: When `with_nms` is True,
            it is tuple[Tensor, Tensor], first tensor bboxes with shape
            [N, num_det, 5], 5 arrange as (x1, y1, x2, y2, score)
            and second element is class labels of shape [N, num_det].
            When `with_nms` is False, first tensor is bboxes with
            shape [N, num_det, 4], second tensor is raw score has
            shape  [N, num_det, num_classes].
        """
        assert len(cls_scores) == len(bbox_preds)

        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]

        assert len(
            img_metas
        ) == 1, 'Only support one input image while in exporting to ONNX'
        img_shape = img_metas[0]['img_shape_for_onnx']

        cfg = self.test_cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_priors)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)

        # e.g. Retina, FreeAnchor, etc.
        if score_factors is None:
            with_score_factors = False
            mlvl_score_factor = [None for _ in range(num_levels)]
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True
            mlvl_score_factor = [
                score_factors[i].detach() for i in range(num_levels)
            ]
            mlvl_score_factors = []

        mlvl_batch_bboxes = []
        mlvl_scores = []

        for cls_score, bbox_pred, score_factors, priors in zip(
                mlvl_cls_scores, mlvl_bbox_preds, mlvl_score_factor,
                mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(0, 2, 3,
                                       1).reshape(batch_size, -1,
                                                  self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = scores.sigmoid()
                nms_pre_score = scores
            else:
                scores = scores.softmax(-1)
                nms_pre_score = scores

            if with_score_factors:
                score_factors = score_factors.permute(0, 2, 3, 1).reshape(
                    batch_size, -1).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            priors = priors.expand(batch_size, -1, priors.size(-1))
            # Get top-k predictions
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:

                if with_score_factors:
                    nms_pre_score = (nms_pre_score * score_factors[..., None])
                else:
                    nms_pre_score = nms_pre_score

                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = nms_pre_score.max(-1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = nms_pre_score[..., :-1].max(-1)
                _, topk_inds = max_scores.topk(nms_pre)

                batch_inds = torch.arange(
                    batch_size, device=bbox_pred.device).view(
                        -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                transformed_inds = bbox_pred.shape[1] * batch_inds + topk_inds
                priors = priors.reshape(
                    -1, priors.size(-1))[transformed_inds, :].reshape(
                        batch_size, -1, priors.size(-1))
                bbox_pred = bbox_pred.reshape(-1,
                                              4)[transformed_inds, :].reshape(
                                                  batch_size, -1, 4)
                scores = scores.reshape(
                    -1, self.cls_out_channels)[transformed_inds, :].reshape(
                        batch_size, -1, self.cls_out_channels)
                if with_score_factors:
                    score_factors = score_factors.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)

            bboxes = self.bbox_coder.decode(
                priors, bbox_pred, max_shape=img_shape)

            mlvl_batch_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if with_score_factors:
                mlvl_score_factors.append(score_factors)

        batch_bboxes = torch.cat(mlvl_batch_bboxes, dim=1)
        batch_scores = torch.cat(mlvl_scores, dim=1)
        if with_score_factors:
            batch_score_factors = torch.cat(mlvl_score_factors, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment

        from mmdet.core.export import add_dummy_nms_for_onnx

        if not self.use_sigmoid_cls:
            batch_scores = batch_scores[..., :self.num_classes]

        if with_score_factors:
            batch_scores = batch_scores * (batch_score_factors.unsqueeze(2))

        if with_nms:
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_bboxes, batch_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        else:
            return batch_bboxes, batch_scores
