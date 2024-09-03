# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mmcv import ops

def norm(feat):
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    feat = (feat - mean) / (std + 1e-6)
    return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True, downsample_stride=2):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(downsample_stride, downsample_stride))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :
        :
        '''

        batch_size = x.size(0)  #   2 , 256 , 300 , 300

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 150 x 150
        g_x = g_x.permute(0, 2, 1)                                  #   2 , 150 x 150, 128

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)   #   2 , 128 , 300 x 300
        theta_x = theta_x.permute(0, 2, 1)                                  #   2 , 300 x 300 , 128
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)       #   2 , 128 , 150 x 150
        f = torch.matmul(theta_x, phi_x)    #   2 , 300x300 , 150x150
        N = f.size(-1)  #   150 x 150
        f_div_C = f / N #   2 , 300x300, 150x150

        y = torch.matmul(f_div_C, g_x)  #   2, 300x300, 128
        y = y.permute(0, 2, 1).contiguous() #   2, 128, 300x300
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)
    return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


def dist2(tensor_a, tensor_b, attention_mask=None, channel_attention_mask=None):
    diff = (tensor_a - tensor_b) ** 2
    #   print(diff.size())      batchsize x 1 x W x H,
    #   print(attention_mask.size()) batchsize x 1 x W x H
    if attention_mask is not None:
        diff = diff * attention_mask
    if channel_attention_mask is not None:
        diff = diff * channel_attention_mask
    diff = torch.sum(diff) ** 0.5
    return diff


def plot_attention_mask(mask):
    mask = torch.squeeze(mask, dim=0)
    mask = mask.cpu().detach().numpy()
    plt.imshow(mask)
    plt.plot(mask)
    plt.savefig('1.png')
    print('saved')
    input()

def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois

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
            return bboxes1.new(batch_shape + (rows, ))
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

        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
        rho2 = left + right
        enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
        enclose_c = enclose_wh[..., 0]**2 + enclose_wh[..., 1]**2
        enclose_c = torch.max(enclose_c, eps)
        dious = ious - rho2 / enclose_c
    return dious

def build_roi_layers(layer_cfg, featmap_strides):
    """Build RoI operator to extract feature from each level feature map.

    Args:
        layer_cfg (dict): Dictionary to construct and config RoI layer
            operation. Options are modules under ``mmcv/ops`` such as
            ``RoIAlign``.
        featmap_strides (List[int]): The stride of input feature map w.r.t
            to the original image size, which would be used to scale RoI
            coordinate (original image coordinate system) to feature
            coordinate system.

    Returns:
        nn.ModuleList: The RoI extractor modules for each level feature
            map.
    """

    cfg = layer_cfg.copy()
    layer_type = cfg.pop('type')
    assert hasattr(ops, layer_type)
    layer_cls = getattr(ops, layer_type)
    roi_layers = nn.ModuleList(
        [layer_cls(spatial_scale=1 / s, **cfg) for s in featmap_strides])
    return roi_layers

def map_roi_levels(rois, num_levels):
    """Map rois to corresponding feature levels by scales.

    - scale < finest_scale * 2: level 0
    - finest_scale * 2 <= scale < finest_scale * 4: level 1
    - finest_scale * 4 <= scale < finest_scale * 8: level 2
    - scale >= finest_scale * 8: level 3

    Args:
        rois (Tensor): Input RoIs, shape (k, 5).
        num_levels (int): Total level number.

    Returns:
        Tensor: Level index (0-based) of each RoI, shape (k, )
    """
    scale = torch.sqrt(
        (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
    target_lvls = torch.floor(torch.log2(scale / 56 + 1e-6))
    target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
    return target_lvls

def generate_correlation_matrix(feat, simf="instance_sim"):
    """
0       :param feat:
    :param similarity_metric:
    :return:
    """
    correlation_matrix_list = []
    for feat_per_image in feat:
        # [M, C]
        if simf == "instance_sim":
            num_instances = feat_per_image.size(0)
            # import pdb;pdb.set_trace()
            # feat_per_image_row = feat_per_image.unsqueeze(2).expand(-1, -1, num_instances)
            # feat_per_image_col = feat_per_image.unsqueeze(2).expand(-1, -1, num_instances).transpose(0, 2)
            # sim = F.cosine_similarity(feat_per_image_row, feat_per_image_col, dim=1)
            feat_per_image = torch.flatten(feat_per_image, 1)
            feat_per_image_normalized = F.normalize(feat_per_image)
            sim = torch.mm(feat_per_image_normalized, feat_per_image_normalized.T)

        if simf == "channel_sim":
            num_instances = feat_per_image.size(0)
            feat_per_image = feat_per_image.view(num_instances, 256, -1)
            # [N, 256, 49]
            feat_per_image_row = feat_per_image.unsqueeze(3).expand(-1, -1, -1, num_instances)
            feat_per_image_col = feat_per_image.unsqueeze(3).expand(-1, -1, -1, num_instances).transpose(0, 3)
            # [N, 256, N]
            sim = F.cosine_similarity(feat_per_image_row, feat_per_image_col, dim=2)
            sim = sim.mean(dim=1)

        if feat_per_image.shape[0] == 0:
            sim = feat_per_image.sum() * 0.
        correlation_matrix_list.append(sim)
    return correlation_matrix_list


def corr_mat_mse_loss(matlist_t, matlist_s, hyper_a,reduction='mean'):
    loss = 0.
    for mat_t, mat_s in zip(matlist_t, matlist_s):
        loss += hyper_a * F.mse_loss(mat_t, mat_s)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    else:
        raise ValueError('must specify reduction as none or mean.')


def rela_batch_aug_two_stage(poposal, relation_batch, positive_batch, threshold=0.5):
    for i in range(len(poposal)):
        if len(positive_batch[i]) == 0:
            # _, max_ind = torch.max(iou, dim=1)
            zero_label = torch.zeros(poposal[i].size(0)).cuda()
            losses = F.binary_cross_entropy(poposal[i][:, -1], zero_label.float(), reduction='none')
            indices = torch.nonzero(losses > threshold).squeeze()
            if indices.numel() == 0:
                continue
            elif indices.numel() == 1:
                indices = indices.unsqueeze(0)
                relation_batch[i] = torch.cat((relation_batch[i], poposal[i][indices][:, :-1]), dim=0)
            else:
                relation_batch[i] = torch.cat((relation_batch[i], poposal[i][indices][:, :-1]), dim=0)
        else:
            mask = torch.ones(poposal[i].size(0), dtype=torch.bool)
            mask[positive_batch[i]] = False
            C = poposal[i][mask]
            if C.shape[0] == 0:
                continue
            else:
                zero_label = torch.zeros(C.size(0)).cuda()
                losses = F.binary_cross_entropy(C[:, -1], zero_label.float(), reduction='none')
                indices = torch.nonzero(losses > threshold).squeeze()
                if indices.numel() == 0:
                    continue
                elif indices.numel() == 1:
                    indices = indices.unsqueeze(0)
                    relation_batch[i] = torch.cat((relation_batch[i], poposal[i][indices][:, :-1]), dim=0)
                else:
                    relation_batch[i] = torch.cat((relation_batch[i], C[indices][:, :-1]), dim=0)
    return relation_batch

# def rela_batch_aug(poposal, relation_batch, positive_batch, iou_batch, threshold=0.5):
#     for i in range(len(poposal)):
#         if len(positive_batch[i]) == 0:
#             _, max_ind = torch.max(iou_batch[i], dim=1)
#             selected_data = torch.index_select(gt_labels, 0, max_ind)
#
#             losses = F.binary_cross_entropy(poposal_cls[i], one_hot_data.float(), reduction='none')
#             indices = torch.nonzero(losses > threshold).squeeze()
#             if indices.size(0) == 0:
#                 continue
#             else:
#                 relation_batch[i] = torch.cat((relation_batch[i], poposal[i][indices][:, :-1]), dim=0)
#         else:
#             mask = torch.ones(poposal[i].size(0), dtype=torch.bool)
#             mask[positive_batch[i]] = False
#             C = poposal[i][mask]
#             if C.shape[0] == 0:
#                 continue
#             else:
#                 zero_label = torch.zeros(C.size(0))
#                 losses = F.binary_cross_entropy(C[:, -1], zero_label.float(), reduction='none')
#                 indices = torch.nonzero(losses > threshold).squeeze()
#                 if indices.size(0) == 0:
#                     continue
#                 else:
#                     relation_batch[i] = torch.cat((relation_batch[i], C[indices][:, :-1]), dim=0)
#     return relation_batch



# def rela_batch_aug(poposal, relation_batch, positive_batch, number):
#     for i in range(len(poposal)):
#         if len(positive_batch[i]) == 0:
#             sorted_indices = torch.argsort(poposal[i][:, -1], descending=True)
#             relation_batch[i] = torch.cat((relation_batch[i], poposal[i][sorted_indices][:number][:, :-1]), dim=0)
#
#         else:
#             mask = torch.ones(poposal[i].size(0), dtype=torch.bool)
#             mask[positive_batch[i]] = False
#             C = poposal[i][mask]
#             if C.shape[0] == 0:
#                 continue
#             elif C.shape[0] > 0 and C.shape[0] < number:
#                 sorted_indices = torch.argsort(C[:, -1], descending=True)
#                 relation_batch[i] = torch.cat((relation_batch[i], C[sorted_indices][:, :-1]), dim=0)
#             else:
#                 sorted_indices = torch.argsort(C[:, -1], descending=True)
#                 relation_batch[i] = torch.cat((relation_batch[i], C[sorted_indices][:number][:, :-1]), dim=0)
#
#     return relation_batch


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        featmap_strides = [8, 16, 32, 64]
        roi_layer = {'type': 'RoIAlign', 'output_size': 7, 'sampling_ratio': 0}
        self.roi_layers = build_roi_layers(roi_layer, featmap_strides)

        self.student_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        self.teacher_non_local = nn.ModuleList(
            [
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=8),
                NonLocalBlockND(in_channels=256, inter_channels=64, downsample_stride=4),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256),
                NonLocalBlockND(in_channels=256)
            ]
        )
        #
        self.adaptation_layers = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        ])

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def relationbatch(self, poposal, ground_turth, thre):
        relation_batch = []
        positive_batch = []
        for i in range(len(poposal)):
            diou = bbox_overlaps(poposal[i][:, :-1], ground_turth[i])
            if diou.shape[0] != poposal[i][:, :-1].shape[0] or diou.shape[1] != ground_turth[i].shape[0] or diou.shape[
                0] == 0 or diou.shape[1] == 0:

                relation_batch.append(ground_turth[i])
                positive_batch.append(torch.empty(0).cuda())
            else:
                max_iou, _ = torch.max(diou, dim=0)
                mask_per_img = torch.zeros([poposal[i][:, :-1].shape[0], 1], dtype=torch.double).cuda()
                for ins in range(ground_turth[i].shape[0]):
                    max_iou_per_gt = max_iou[ins] * thre
                    mask_per_gt = diou[:, ins] > max_iou_per_gt
                    mask_per_gt = mask_per_gt.float().unsqueeze(-1)
                    mask_per_img += mask_per_gt

                mask_per_img = mask_per_img.squeeze(-1).nonzero().squeeze(-1)
                positive_batch.append(mask_per_img)
                selected_rows = poposal[i][:, :-1][mask_per_img]

                if selected_rows.size()[0] == 0:
                    relation_batch.append(ground_turth[i])
                elif selected_rows.size()[0] < ground_turth[i].size()[0]:
                    relation_batch.append(torch.cat((selected_rows, ground_turth[i]), 0))
                else:
                    relation_batch.append(selected_rows)

        return relation_batch, positive_batch, diou

    def roifeat(self, feats, rois):

        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)

        roi_feats = feats[0].new_zeros(
            rois.size(0), 256, *out_size)

        target_lvls = map_roi_levels(rois, num_levels)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                roi_feats += feats[i].sum() * 0.
        return roi_feats

    def get_teacher_info(self,
                         img,
                         img_metas,
                         gt_bboxes,
                         gt_labels,
                         gt_bboxes_ignore=None,
                         gt_masks=None,
                         proposals=None,
                         t_feats=None,
                         **kwargs):
        teacher_info = {}
        x = self.extract_feat(img)
        tea_bbox_outs = self.bbox_head(x)
        # proposal_cfg = self.train_cfg.get('rpn_proposal',
        #                                   self.test_cfg.rpn)
        proposal_cfg = self.train_cfg.get('rpn_proposal_KD',
                                          self.test_cfg)
        # print(proposal_cfg)

        teacher_info.update({'feat': x, "tea_bbox_outs": tea_bbox_outs, "tea_proposal_cfg": proposal_cfg})
        # RPN forward and loss

        return teacher_info





    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      t_info=None,
                      epoch=None,
                      iter=None,
                      ):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        t_feats = t_info['feat']
        #tea_proposal_cfg = t_info['tea_proposal_cfg']
        tea_bbox_outs = t_info['tea_bbox_outs']
        stu_bbox_outs = self.bbox_head(x)
        tea_proposal_cfg = self.train_cfg.get('rpn_proposal_KD',
                                          self.test_cfg)
        stu_proposal= self.bbox_head.get_bboxes_KD_single(*stu_bbox_outs, img_metas=img_metas,
                                                   cfg=tea_proposal_cfg)
        # for i in range(2):
        #     print(stu_tea_proposal[i].size())
        stu_tea_proposal = []
        for i in range(len(stu_proposal)):
            stu_tea_proposal.append(stu_proposal[i][0])


        # for i in range(2):
        #     print(stu_tea_proposal[i].size())
        #     print(stu_proposal_gtlabels[i].size())
        # stu_tea_proposal = []
        #
        # for i in range(len(stu_proposal)):
        #     stu_tea_proposal.append(stu_proposal[i][0])

        relation_batch, positive_batch, iou = self.relationbatch(poposal=stu_tea_proposal, ground_turth=gt_bboxes, thre=0.6)

        relation_batch = rela_batch_aug_two_stage(poposal=stu_tea_proposal, relation_batch=relation_batch, positive_batch=positive_batch, threshold=0.5)

        rois = bbox2roi([res for res in relation_batch])

        tea_roifeats = self.roifeat(t_feats[:4], rois)
        stu_roifeats = self.roifeat(x[:4], rois)

        tea_roifeats = torch.split(tea_roifeats, [rela.size()[0] for rela in relation_batch])
        stu_roifeats = torch.split(stu_roifeats, [rela.size()[0] for rela in relation_batch])

        teacher_region_correlation_matrices_pool = generate_correlation_matrix(tea_roifeats)
        student_region_correlation_matrices_pool = generate_correlation_matrix(stu_roifeats)

        kd_inrelation_loss = corr_mat_mse_loss(teacher_region_correlation_matrices_pool,
                                               student_region_correlation_matrices_pool, hyper_a=0.5)

        losses.update({'kd_inrelation_loss': kd_inrelation_loss})


        kd_nonlocal_loss = 0
        if t_info is not None:
            for _i in range(len(t_feats)):
                s_relation = self.student_non_local[_i](x[_i])
                t_relation = self.teacher_non_local[_i](t_feats[_i])
                #   print(s_relation.size())
                #kd_nonlocal_loss += torch.dist(self.adaptation_layers[_i](s_relation), t_relation, p=2)
                kd_nonlocal_loss += torch.dist(self.adaptation_layers[_i](s_relation), t_relation, p=2)

        losses.update({'kd_nonlocal_loss':kd_nonlocal_loss * 7e-5})

        stu_cls_score = stu_bbox_outs[0]
        tea_cls_score = tea_bbox_outs[0]
        stu_reg_score = stu_bbox_outs[1]
        tea_reg_score = tea_bbox_outs[1]
        distill_cls_loss = 0
        distill_cls_weight = 0.06
        #
        for layer in range(len(stu_cls_score)):
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid()
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid().detach()
            score = torch.abs(tea_cls_score_sigmoid - stu_cls_score_sigmoid)
            mask = torch.max(score, dim=1).values
            mask = mask.detach()

            cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid, reduction='none')

            distill_cls_loss += (cls_loss * mask[:, None, :, :]).sum() / mask.sum()
            # distill_cls_loss += pearson_loss_with_weights(stu_cls_score_sigmoid , tea_cls_score_sigmoid,mask[:, None, :, :])
            # distill_cls_loss  += pearson_loss(stu_cls_score_sigmoid * mask[:, None, :, :], tea_cls_score_sigmoid * mask[:, None, :, :])
        distill_cls_loss = distill_cls_loss * distill_cls_weight
        losses.update({'distill_cls_loss': distill_cls_loss})

        loss_reg = self.bbox_head.reg_distill_single_fcos(stu_reg=stu_reg_score, tea_reg=tea_reg_score, tea_cls=tea_cls_score,
                                             stu_cls=stu_cls_score, gt_truth=gt_bboxes, img_metas=img_metas)

        losses.update({'distill_reg_loss': loss_reg * 1.0})





        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels

