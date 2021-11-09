from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.cnn import constant_init
from mmdet.core import (PointGenerator, multi_apply, multiclass_rnms,
                       levels_to_images)
from mmdet.ops import ConvModule, DeformConv
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob
from mmdet.core.bbox import init_pointset_target, refine_pointset_target
from mmdet.ops.minarearect import minaerarect
from mmdet.ops.chamfer_distance import ChamferDistance2D
import math

@HEADS.register_module
class OrientedRepPointsHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_rbox_init=dict(
                     type='IoULoss', loss_weight=0.4),
                 loss_rbox_refine=dict(
                     type='IoULoss', loss_weight=0.75),
                 loss_spatial_init=dict(type='SpatialBorderLoss', loss_weight=0.05),
                 loss_spatial_refine=dict(type='SpatialBorderLoss', loss_weight=0.1),
                 center_init=True,
                 top_ratio=0.4):

        super(OrientedRepPointsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_points = num_points
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.loss_cls = build_loss(loss_cls)
        self.loss_rbox_init = build_loss(loss_rbox_init)
        self.loss_rbox_refine = build_loss(loss_rbox_refine)
        self.loss_spatial_init = build_loss(loss_spatial_init)
        self.loss_spatial_refine = build_loss(loss_spatial_refine)
        self.center_init = center_init
        self.top_ratio = top_ratio

        self.count_batch = 0
        self.count_gt = 0
        self.count_pos_point_init = 0
        self.count_pos_point_quality_assess = 0

        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self.point_generators = [PointGenerator() for _ in self.point_strides]
        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            
        pts_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = DeformConv(self.feat_channels,
                                             self.point_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels,
                                           self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels,
                                                 self.point_feat_channels, 3,
                                                 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels,
                                                pts_out_dim, 1, 1, 0)
        self.reppoints_pts_refine_conv = DeformConv(self.feat_channels,
                                                    self.point_feat_channels,
                                                    self.dcn_kernel, 1,
                                                    self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels,
                                                  pts_out_dim, 1, 1, 0)
        
    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
            
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)

    def forward_single(self, x):  # x： 5*tensor:  tensor.size(b, c, f_h, f_w)
        dcn_base_offset = self.dcn_base_offset.type_as(x)  # (1, 18, 1, 1) 3*3卷积层的base坐标
        points_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)                  # 分类分支(b, 256, f_h, f_w)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)                  # 语义点分支(b, 256, f_h, f_w)
        # initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(        # points_init分支： pts_out_init.size(b, 18, f_h, f_w)  每个特征点生成9个语义点
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init
        # refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init   # pts_out_init分支梯度乘上系数
        dcn_offset = pts_out_init_grad_mul - dcn_base_offset         # 预测的9个语义点坐标pts_out_init减去base坐标得到dcn的offset
        dcn_cls_feat = self.reppoints_cls_conv(cls_feat, dcn_offset) # dcn refine分类分支cls_feat
        cls_out = self.reppoints_cls_out(self.relu(dcn_cls_feat))    # dcn refine分类分支cls_feat
        pts_out_refine = self.reppoints_pts_refine_out(self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))

        pts_out_refine = pts_out_refine + pts_out_init.detach()      # refine后的语义点预测值 + 初始语义点预测值 = 最后的9个语义点坐标
        return cls_out, pts_out_init, pts_out_refine 

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_points(self, featmap_sizes, img_metas):
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)
        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(    # points.size(f_h*f_w, [x, y, stride]]) eg: [1000, 1016, 8]
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)                 # multi_level_points: list[5 * points]
        points_list = [[point.clone() for point in multi_level_points]  # points_list : list[b * multi_level_points]
                       for _ in range(num_imgs)]
        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return points_list, valid_flag_list        # points_list : list[b * multi_level_points]  valid_flag_list：[b * multi_level_flags]

    def offset_to_pts(self, center_list, pred_list):
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)

                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def neargtcorner(self, pts, gtbboxes):
        gtbboxes = gtbboxes.view(-1, 4, 2)
        pts = pts.view(-1, self.num_points, 2)

        pts_corner_first_ind = ((gtbboxes[:, 0:1, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_first_ind = pts_corner_first_ind.reshape(pts_corner_first_ind.shape[0], 1, 1).expand(-1, -1,
                                                                                                        pts.shape[2])
        pts_corner_first = torch.gather(pts, 1, pts_corner_first_ind).squeeze(1)

        pts_corner_sec_ind = ((gtbboxes[:, 1:2, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_sec_ind = pts_corner_sec_ind.reshape(pts_corner_sec_ind.shape[0], 1, 1).expand(-1, -1, pts.shape[2])
        pts_corner_sec = torch.gather(pts, 1, pts_corner_sec_ind).squeeze(1)

        pts_corner_third_ind = ((gtbboxes[:, 2:3, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_third_ind = pts_corner_third_ind.reshape(pts_corner_third_ind.shape[0], 1, 1).expand(-1, -1,
                                                                                                        pts.shape[2])
        pts_corner_third = torch.gather(pts, 1, pts_corner_third_ind).squeeze(1)

        pts_corner_four_ind = ((gtbboxes[:, 3:4, :] - pts) ** 2).sum(dim=2).min(dim=1)[1]
        pts_corner_four_ind = pts_corner_four_ind.reshape(pts_corner_four_ind.shape[0], 1, 1).expand(-1, -1,
                                                                                                     pts.shape[2])
        pts_corner_four = torch.gather(pts, 1, pts_corner_four_ind).squeeze(1)

        corners = torch.cat([pts_corner_first, pts_corner_sec, pts_corner_third, pts_corner_four], dim=1)
        return corners

    def sampling_points(self, corners, points_num):  
        """
        Args:
            corners(tensor) : torch.size(n, 8), 四边形的四个点的位置
            points_num(int) : 每条边上的采样点数量
        return:
            all_points(tensor) : torch.size(n, 4*points_num, 2) ，四边形的采样点集的绝对坐标
        """
        device = corners.device
        corners_xs, corners_ys = corners[:, 0::2], corners[:, 1::2]
        first_edge_x_points = corners_xs[:, 0:2]  # 第一条边取x坐标 (n, 2)
        first_edge_y_points = corners_ys[:, 0:2]  # 第一条边取y坐标 (n, 2)
        sec_edge_x_points = corners_xs[:, 1:3]
        sec_edge_y_points = corners_ys[:, 1:3]
        third_edge_x_points = corners_xs[:, 2:4]
        third_edge_y_points = corners_ys[:, 2:4]
        four_edge_x_points_s = corners_xs[:, 3]
        four_edge_y_points_s = corners_ys[:, 3]
        four_edge_x_points_e = corners_xs[:, 0]
        four_edge_y_points_e = corners_ys[:, 0]

        edge_ratio = torch.linspace(0, 1, points_num).to(device).repeat(corners.shape[0], 1)  # 0-1采样points_num个ratio，并重复n次  torch.size(n, points_num)
        all_1_edge_x_points = edge_ratio * first_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * first_edge_x_points[:, 0:1]   # (n, points_num)  开始间隔采样，得到真实坐标
        all_1_edge_y_points = edge_ratio * first_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * first_edge_y_points[:, 0:1]

        all_2_edge_x_points = edge_ratio * sec_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * sec_edge_x_points[:, 0:1]
        all_2_edge_y_points = edge_ratio * sec_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * sec_edge_y_points[:, 0:1]

        all_3_edge_x_points = edge_ratio * third_edge_x_points[:, 1:2] + \
                              (1 - edge_ratio) * third_edge_x_points[:, 0:1]
        all_3_edge_y_points = edge_ratio * third_edge_y_points[:, 1:2] + \
                              (1 - edge_ratio) * third_edge_y_points[:, 0:1]

        all_4_edge_x_points = edge_ratio * four_edge_x_points_e.unsqueeze(1) + \
                              (1 - edge_ratio) * four_edge_x_points_s.unsqueeze(1)
        all_4_edge_y_points = edge_ratio * four_edge_y_points_e.unsqueeze(1) + \
                              (1 - edge_ratio) * four_edge_y_points_s.unsqueeze(1)

        all_x_points = torch.cat([all_1_edge_x_points, all_2_edge_x_points,      # (n, 4*points_num, 1)
                                  all_3_edge_x_points, all_4_edge_x_points], dim=1).unsqueeze(dim=2)

        all_y_points = torch.cat([all_1_edge_y_points, all_2_edge_y_points,      # (n, 4*points_num, 1)
                                  all_3_edge_y_points, all_4_edge_y_points], dim=1).unsqueeze(dim=2)

        all_points = torch.cat([all_x_points, all_y_points], dim=2)              # (n, 4*points_num, 2)
        return all_points

    def init_loss_single(self,  pts_pred_init, rbox_gt_init, rbox_weights_init, stride):

        normalize_term = self.point_base_scale * stride
        rbox_gt_init = rbox_gt_init.reshape(-1, 8)
        rbox_weights_init = rbox_weights_init.reshape(-1)
        pts_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        pos_ind_init = (rbox_weights_init > 0).nonzero().reshape(-1)
        pts_pred_init_norm = pts_pred_init[pos_ind_init]
        rbox_gt_init_norm = rbox_gt_init[pos_ind_init]
        rbox_weights_pos_init = rbox_weights_init[pos_ind_init]
        loss_rbox_init = self.loss_rbox_init(
            pts_pred_init_norm / normalize_term,
            rbox_gt_init_norm / normalize_term,
            rbox_weights_pos_init
        )

        loss_border_init = self.loss_spatial_init(
            pts_pred_init_norm.reshape(-1, 2 * self.num_points) / normalize_term,
            rbox_gt_init_norm / normalize_term,
            rbox_weights_pos_init,
            y_first=False,
            avg_factor=None
        ) if self.loss_spatial_init is not None else loss_rbox_init.new_zeros(1)

        return loss_rbox_init, loss_border_init

    def loss(self,
             cls_scores,       # List[5*tesnor]  tensor.size = (b, num_classes, f_h, f_w)
             pts_preds_init,   # List[5*tesnor]  tensor.size = (b, 18, f_h, f_w)
             pts_preds_refine, # List[5*tesnor]  tensor.size = (b, 18, f_h, f_w)
             gt_rbboxes,       # List[b*tensor]  tensor.size = (n, 8)
             gt_labels,        # List[b*tensor]  tensor.size = (n, )
             img_metas,        # List[b*dict]
             cfg,
             gt_rbboxes_ignore=None):

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        pts_coordinate_preds_init = self.offset_to_pts(center_list,   # init stage语义点预测的真实坐标 list[5*tensor] tensor.size(b, f_h*f_w, 18)
                                                       pts_preds_init)

        num_proposals_each_level = [(featmap.size(-1) * featmap.size(-2))  # size(5) f_h*f_w
                                    for featmap in cls_scores]

        num_level = len(featmap_sizes)
        assert num_level == len(pts_coordinate_preds_init)
        candidate_list = center_list

        #init_stage assign
        cls_reg_targets_init = init_pointset_target(
            candidate_list,
            valid_flag_list,
            gt_rbboxes,
            img_metas,
            cfg.init,
            gt_rbboxes_ignore_list=gt_rbboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        # get the number of sample of assign)
        (*_, rbbox_gt_list_init, candidate_list_init, rbox_weights_list_init,
         num_total_pos_init, num_total_neg_init, gt_inds_init) = cls_reg_targets_init

        # target for refinement stage
        center_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        pts_coordinate_preds_refine = self.offset_to_pts(center_list, pts_preds_refine)  # refine阶段语义点预测的真实坐标 list[5*tensor] tensor.size(b, f_h*f_w, 18)

        points_list = []
        for i_img, center in enumerate(center_list): # 遍历每张图
            points = []
            for i_lvl in range(len(pts_preds_refine)): # 遍历每个尺度feature
                points_preds_init_ = pts_preds_init[i_lvl].detach()
                points_preds_init_ = points_preds_init_.view(points_preds_init_.shape[0], -1,
                                                             *points_preds_init_.shape[2:])
                points_shift = points_preds_init_.permute(0, 2, 3, 1) * self.point_strides[i_lvl]
                points_center = center[i_lvl][:, :2].repeat(1, self.num_points)
                points.append(points_center + points_shift[i_img].reshape(-1, 2 * self.num_points))
            points_list.append(points)   # list[b*tensor] tensor.size(-1, 18) 原图上所有point样本的真实坐标

        cls_reg_targets_refine = refine_pointset_target(
            points_list,
            valid_flag_list,
            gt_rbboxes,
            img_metas,
            cfg.refine,
            gt_rbboxes_ignore_list=gt_rbboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)

        (labels_list, label_weights_list, rbox_gt_list_refine,
         _, rbox_weights_list_refine, pos_inds_list_refine,    # pos_inds_list_refine list[b*tensor] tensor.size(topk*num_gt) 正样本point的索引
         pos_gt_index_list_refine) = cls_reg_targets_refine    # pos_gt_index_list_refine list[b*tensor] tensor.size(num_gt) 与正样本匹配的gt的索引（1~num_gt）
 
        cls_scores = levels_to_images(cls_scores) # List[5*tesnor]  tensor.size = (b, num_classes, f_h, f_w)  -> List[b*tensor] tensor.size(-1, num_classes)
        cls_scores = [
            item.reshape(-1, self.cls_out_channels) for item in cls_scores
        ]

        pts_coordinate_preds_init_image = levels_to_images( # List[5*tesnor]  tensor.size = (b, f_h*f_w, 18)  -> List[b*tensor] tensor.size(-1, 18)
            pts_coordinate_preds_init, flatten=True)
        pts_coordinate_preds_init_image = [
            item.reshape(-1, 2 * self.num_points) for item in pts_coordinate_preds_init_image
        ]

        pts_coordinate_preds_refine_image = levels_to_images(  # List[5*tesnor]  tensor.size = (b, f_h*f_w, 18)  -> List[b*tensor] tensor.size(-1, 18)
            pts_coordinate_preds_refine, flatten=True)
        pts_coordinate_preds_refine_image = [
            item.reshape(-1, 2 * self.num_points) for item in pts_coordinate_preds_refine_image
        ]

        with torch.no_grad():

            # refine_stage loc loss
            # pos_losses_list, = multi_apply(self.points_quality_assessment, cls_scores,
            #                                pts_coordinate_preds_refine_image, labels_list,
            #                                rbox_gt_list_refine, label_weights_list,
            #                                rbox_weights_list_refine, pos_inds_list_refine)

            # init stage and refine stage loc loss 计算初始阶段正样本的质量评估值
            quality_assess_list, = multi_apply(self.points_quality_assessment, cls_scores,   # List[b*tensor]  tensor.size(gt_label_num*topk)
                                           pts_coordinate_preds_init_image, pts_coordinate_preds_refine_image, labels_list,
                                           rbox_gt_list_refine, label_weights_list,
                                           rbox_weights_list_refine, pos_inds_list_refine)


            labels_list, label_weights_list, rbox_weights_list_refine, num_pos, pos_normalize_term = multi_apply(
                self.point_samples_selection,
                quality_assess_list,
                labels_list,
                label_weights_list,
                rbox_weights_list_refine,
                pos_inds_list_refine,
                pos_gt_index_list_refine,
                num_proposals_each_level=num_proposals_each_level,
                num_level=num_level
            )
            num_pos = sum(num_pos)

        cls_scores = torch.cat(cls_scores, 0).view(-1, cls_scores[0].size(-1))   # List[b*tensor] tensor.size(-1, num_classes) -> tensor.size(-1, num_classes)
        pts_preds_refine = torch.cat(pts_coordinate_preds_refine_image,          # List[b*tensor] tensor.size(-1, 18)  -> tensor.size(-1, 18)
                                     0).view(-1, pts_coordinate_preds_refine_image[0].size(-1))
        labels = torch.cat(labels_list, 0).view(-1)                              # List[b*tensor] tensor.size(f_h*f_w)  -> tensor.size(-1)
        labels_weight = torch.cat(label_weights_list, 0).view(-1)
        rbox_gt_refine = torch.cat(rbox_gt_list_refine,                          # List[b*tensor] tensor.size(f_h*f_w, 8) -> tensor.size(-1, 8)
                                    0).view(-1, rbox_gt_list_refine[0].size(-1))
        rbox_weights_refine = torch.cat(rbox_weights_list_refine, 0).view(-1)
        pos_normalize_term = torch.cat(pos_normalize_term, 0).reshape(-1)        # tensor.size(num_pos) 重采样后正样本所对应的尺度
        pos_inds_flatten = (labels > 0).nonzero().reshape(-1)                    # tensor.size(num_pos) 本次batch的正样本索引
        assert len(pos_normalize_term) == len(pos_inds_flatten)
        if num_pos:
            losses_cls = self.loss_cls(       # focal loss . cls_scores为整个batch内所有point的类别通道预测值，labels为所有point的类别标签
                cls_scores, labels, labels_weight, avg_factor=num_pos)
            pos_pts_pred_refine = pts_preds_refine[pos_inds_flatten]  # 取出对应正样本的refine阶段的pts预测结果 torch.size(num_pos, 18)
            pos_rbox_gt_refine = rbox_gt_refine[pos_inds_flatten]     # 其匹配的gt真实坐标 torch.size(num_pos, 8)
            pos_rbox_weights_refine = rbox_weights_refine[pos_inds_flatten]
            losses_rbox_refine = self.loss_rbox_refine(   # convex GIOU Loss
                pos_pts_pred_refine / pos_normalize_term.reshape(-1, 1), 
                pos_rbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_rbox_weights_refine
            )
            # SpatialBorderLoss 
            loss_border_refine = self.loss_spatial_refine(  # SpatialBorderLoss 给语义点添加空间限制，惩罚落在gt区域外的语义点
                pos_pts_pred_refine.reshape(-1, 2 * self.num_points) / pos_normalize_term.reshape(-1, 1), 
                pos_rbox_gt_refine / pos_normalize_term.reshape(-1, 1),
                pos_rbox_weights_refine,
                y_first=False,
                avg_factor=None
            ) if self.loss_spatial_refine is not None else losses_rbox_refine.new_zeros(1)

        else:
            losses_cls = cls_scores.sum() * 0
            losses_rbox_refine = pts_preds_refine.sum() * 0
            loss_border_refine = pts_preds_refine.sum() * 0

        losses_rbox_init, loss_border_init = multi_apply(
            self.init_loss_single,
            pts_coordinate_preds_init,
            rbbox_gt_list_init,
            rbox_weights_list_init,
            self.point_strides)

        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_rbox_init': losses_rbox_init,
            'loss_rbox_refine': losses_rbox_refine,
            'loss_spatial_init': loss_border_init,
            'loss_spatial_refine': loss_border_refine
        }
        
        # gt_num = 0
        # for gt_label_img in gt_labels:  # 计算每次batch中的gt数量
        #     gt_num += len(gt_label_img)
        # self.count_batch += 1
        # self.count_gt += gt_num
        # self.count_pos_point_init += num_total_pos_init
        # self.count_pos_point_quality_assess += num_pos
        #print('iter %d, gt_num %d, pos_point_init_num %d, quality_assess_pos_num %d' % (self.count_batch, gt_num, num_total_pos_init, num_pos))
        # if self.count_batch % 50 == 0:
        #     avg_gt_num_perbatch = self.count_gt / self.count_batch
        #     avg_pos_init_num = self.count_pos_point_init / self.count_batch
        #     avg_pos_point_quality_assess = self.count_pos_point_quality_assess / self.count_batch
        #     filename = 'pos_record.txt'
        #     with open(filename, 'a') as file_object:
        #         file_object.write('avg_gt_num_perbatch:' + str(avg_gt_num_perbatch) + ' ' + 'avg_pos_init_num:' + str(avg_pos_init_num) + ' ' + 'avg_pos_point_quality_assess:' + str(avg_pos_point_quality_assess) + '\n')
        #         pass
            
        return loss_dict_all

    def points_quality_assessment(self, cls_score, pts_pred_init, pts_pred_refine, label, rbbox_gt, label_weight, rbox_weight, pos_inds):
        # 评估方式： 分类loss + 定位loss + 方向对齐loss    return qua.size(gt_label_num*topk)
        pos_scores = cls_score[pos_inds]                # (n, class_nums) -> (gt_label_num*topk, class_nums)
        pos_pts_pred_init = pts_pred_init[pos_inds]     # (n, 18) -> (gt_label_num*topk, 18)
        pos_pts_pred_refine = pts_pred_refine[pos_inds] # (n, 18) -> (gt_label_num*topk, 18)

        pos_rbbox_gt = rbbox_gt[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_rbox_weight = rbox_weight[pos_inds]

        qua_cls = self.loss_cls(  # Q_cls_loss  tensor.size(gt_label_num*topk, 18)
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')  # loss先不进行sum求和

        corners_pred_init = minaerarect(pos_pts_pred_init)      # tensor.size(gt_label_num*topk, 18)  -> tensor.size(gt_label_num*topk, 8) 
        corners_pred_refine = minaerarect(pos_pts_pred_refine)  # tensor.size(gt_label_num*topk, 18)  -> tensor.size(gt_label_num*topk, 8) 
        # corners_pred = self.neargtcorner(pos_pts_pred, pos_rbbox_gt)

        sampling_pts_pred_init = self.sampling_points(corners_pred_init, 10)
        sampling_pts_pred_refine = self.sampling_points(corners_pred_refine, 10)
        corners_pts_gt = self.sampling_points(pos_rbbox_gt, 10)
        # torch.size(gt_label_num)
        qua_ori_init = ChamferDistance2D(corners_pts_gt, sampling_pts_pred_init)    # 求gt40个点与初始点集的最小外界矩形框的40个采样点计算倒角距离
        qua_ori_refine = ChamferDistance2D(corners_pts_gt, sampling_pts_pred_refine)# 求gt40个点与refine点集的最小外界矩形框的40个采样点计算倒角距离

        qua_loc_init = self.loss_rbox_refine(  # Q_loc_loss_init: torch.size(gt_label_num*topk)  
            pos_pts_pred_init,
            pos_rbbox_gt,
            pos_rbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        qua_loc_refine = self.loss_rbox_refine(# Q_loc_loss_refine: torch.size(gt_label_num*topk)
            pos_pts_pred_refine,
            pos_rbbox_gt,
            pos_rbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        qua_cls = qua_cls.sum(-1)    # tensor.size(gt_label_num*topk, 18) -> tensor.size(gt_label_num*topk)
        # weight inti-stage and refine-stage
        qua = qua_cls + 0.2*(qua_loc_init + 0.3 * qua_ori_init) + 0.8 * (
                    qua_loc_refine + 0.3 * qua_ori_refine)
        # only refine-stage
        # qua = qua_cls + qua_loc_refine + 0.3 * qua_ori_refine
        return qua,      # torch.size(gt_label_num*topk)

    def point_samples_selection(self, quality_assess, label, label_weight, rbox_weight,   # pos_inds list[b*tensor] tensor.size(topk*num_gt) 正样本point的索引
                     pos_inds, pos_gt_inds, num_proposals_each_level=None, num_level=None): # pos_gt_inds list[b*tensor] tensor.size(num_gt) 与正样本匹配的gt的索引（1~num_gt）
        '''   基于初始正样本的质量评估分数来进行正样本的重新采样 质量低的正样本重新标记为背景
              The selection of point set samples based on the quality assessment values.
        '''

        if len(pos_inds) == 0:
            return label, label_weight, rbox_weight, 0, torch.tensor([]).type_as(rbox_weight)

        num_gt = pos_gt_inds.max()
        num_proposals_each_level_ = num_proposals_each_level.copy()
        num_proposals_each_level_.insert(0, 0)   # 
        inds_level_interval = np.cumsum(num_proposals_each_level_)
        pos_level_mask = []
        for i in range(num_level):
            mask = (pos_inds >= inds_level_interval[i]) & (   # 每个尺度设置一个mask，pos_inds中处于对应尺度上的points索引才为true
                    pos_inds < inds_level_interval[i + 1])
            pos_level_mask.append(mask)
        pos_inds_after_select = []
        ignore_inds_after_select = []

        for gt_ind in range(num_gt):  # 遍历每一个gt
            pos_inds_select = []
            pos_loss_select = []
            gt_mask = pos_gt_inds == (gt_ind + 1)  # 找到该gt的索引，mask对应位置置为True
            for level in range(num_level):  # 遍历每一个feature尺度
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask  # 找到该尺度上gt与匹配的正样本point同时存在的位置，对应mask置为true
                value, topk_inds = quality_assess[level_gt_mask].topk(  # 取出能够匹配该gt的points正样本的质量评估分数，并topk倒序排序,最多topk=6
                    min(level_gt_mask.sum(), 6), largest=False)
                pos_inds_select.append(pos_inds[level_gt_mask][topk_inds])  # 将topk匹配程度最高的points正样本索引 append 进 pos_inds_select
                pos_loss_select.append(value)
            pos_inds_select = torch.cat(pos_inds_select)  # torch.size(n) 该gt在所有feature尺度中正样本points的索引
            pos_loss_select = torch.cat(pos_loss_select)

            if len(pos_inds_select) < 2:  
                pos_inds_after_select.append(pos_inds_select)
                ignore_inds_after_select.append(pos_inds_select.new_tensor([]))
            else:
                pos_loss_select, sort_inds = pos_loss_select.sort() # small to large  每个gt的points正样本进行质量评估分数的排序
                pos_inds_select = pos_inds_select[sort_inds]
                topk = math.ceil(pos_loss_select.shape[0] * self.top_ratio) # topk向上取整
                pos_inds_select_topk = pos_inds_select[:topk]               #
                pos_inds_after_select.append(pos_inds_select_topk)          # List[num_gt*tensor] tensor.size(n) n为质量评估后的topk正样本数量
                ignore_inds_after_select.append(pos_inds_select_topk.new_tensor([]))

        pos_inds_after_select = torch.cat(pos_inds_after_select)  # # List[num_gt*tensor] tensor.size(n)  -> tensor.size(num_gt*tensor) 
        ignore_inds_after_select = torch.cat(ignore_inds_after_select)
        # all(1)判断其每行是否都为True tensor.size(topk*num_gt)  即寻找pos_inds中没被重采样到（被select忽略）的正样本索引
        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_select).all(1)  # tensor.size(topk*num_gt, after_assessment_num) pos_inds与pos_inds_after_select的元素对应情况
        reassign_ids = pos_inds[reassign_mask]  # 质量评估差的正样本point索引
        label[reassign_ids] = 0                 # 将原本的正样本类别置为背景
        # label[reassign_ids] = self.num_classes
        label_weight[ignore_inds_after_select] = 0
        rbox_weight[reassign_ids] = 0
        num_pos = len(pos_inds_after_select)

        pos_level_mask_after_select = []
        for i in range(num_level):    # 遍历每一个feature尺度
            mask = (pos_inds_after_select >= inds_level_interval[i]) & (   # 每个尺度设置一个mask，pos_inds_after_select 中处于对应尺度上的points索引才为true
                    pos_inds_after_select < inds_level_interval[i + 1])
            pos_level_mask_after_select.append(mask)
        pos_level_mask_after_select = torch.stack(pos_level_mask_after_select, 0).type_as(label)
        pos_normalize_term = pos_level_mask_after_select * (
                self.point_base_scale *
                torch.as_tensor(self.point_strides).type_as(label)).reshape(-1, 1)
        pos_normalize_term = pos_normalize_term[pos_normalize_term > 0].type_as(rbox_weight)
        assert len(pos_normalize_term) == len(pos_inds_after_select)

        return label, label_weight, rbox_weight, num_pos, pos_normalize_term  # pos_normalize_term： torch.size(num_pos) 重采样后正样本的对应的尺度

    def get_bboxes(self,
                   cls_scores,                           # cls分支预测输出  list[tensor*5]  torch.size(batch, num_class, H, W)
                   pts_preds_init,                       # points_init分支预测输出  list[tensor*5]  torch.size(batch, points_num*2, H, W)
                   pts_preds_refine,                     # points_refine分支预测输出  list[tensor*5]  torch.size(batch, points_num*2, H, W)
                   img_metas,
                   cfg,
                   rescale=False,
                   nms=True):
        assert len(cls_scores) == len(pts_preds_refine)
        
        num_levels = len(cls_scores)
        mlvl_points = [                                  # list[tensor*5]  torch.size(H*W, 3)   3 - [x y stride]
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [                           # 从class预测分支取对应图像的分类预测结果 list[tensor*5] torch.size(num_class, H, W)
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            points_pred_list = [                         # 从point_refine预测分支取对应图像的points预测结果 list[tensor*5] torch.size(points_num*2, H, W)
                pts_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, points_pred_list,   # proposal: list[det_bboxes, det_labels]
                                               mlvl_points, img_shape,
                                               scale_factor, cfg, rescale, nms)
            result_list.append(proposals)
        return result_list   # list(batch_size * list[det_bboxes, det_labels])  det_bboxes:(n, [18, 8, score]), det_labels:(n)                                               

    def get_bboxes_single(self,
                          cls_scores,                   # 单张图像的class预测结果  list[tensor*5]   torch.size(num_class, H, W)
                          points_preds,                 # 单张图像的points_refine预测结果  list[tensor*5]  torch.size(points_num*2, H, W)
                          mlvl_points,                  # points生成列表  list[tensor*5]  torch.size(H*W, 3)   3 - [x y stride]
                          img_shape,                    # 图像shape
                          scale_factor,                 # 图像相比于原图的比例
                          cfg,                          # test_cfg
                          rescale=False,
                          nms=True):
        assert len(cls_scores) == len(points_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_reppoints = []
        
        for i_lvl, (cls_score, points_pred, points) in enumerate(               # 遍历每个尺度
                zip(cls_scores, points_preds, mlvl_points)):
            assert cls_score.size()[-2:] == points_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,                                          # cls_score tensor   torch.size(num_class, H, W) -> torch.size(H*W, num_class) 
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            points_pred = points_pred.permute(1, 2, 0).reshape(-1, 2 * self.num_points)  # points_pred tensor   torch.size(points_num*2, H, W) -> torch.size(H*W, points_num*2)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)                                    # 获取当前points的预测类别分数tensor(H*W) 和   类别tensor(H*W)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]                                            # 保留topk的points的真实位置
                points_pred = points_pred[topk_inds, :]                                  # 保留topk的points预测结果
                scores = scores[topk_inds, :]                                            # 保留topk的scores预测结果

            pts_pred = points_pred.reshape(-1, self.num_points, 2)                       # pts_pred  torch.size(num_topk, points_num, 2)
            pts_pred_offsety = pts_pred[:, :, 0::2]                                      # pts_pred_y  torch.size(num_topk, points_num, 1)  y_first
            pts_pred_offsetx = pts_pred[:, :, 1::2]                                      # pts_pred_x  torch.size(num_topk, points_num, 1)
            pts_pred = torch.cat([pts_pred_offsetx, pts_pred_offsety], dim=2).reshape(-1, 2 * self.num_points) # pts_pred  torch.size(num_topk, points_num*2) x_first
            bbox_pred = minaerarect(pts_pred)                                            # bbox_pred   torch.size(num_topk, 8)   9points -> 1rbox

            bbox_pos_center = points[:, :2].repeat(1, 4)                                 # 找points中心的真实坐标 重复4次以便和预测点转换后的rbox相加得到真实坐标
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            
            mlvl_bboxes.append(bboxes)                                                   # 本尺度预测的真实bboxes append进mlvl_bboxes 列表
            mlvl_scores.append(scores)                                                   # ...

            points_pred = points_pred.reshape(-1, self.num_points, 2)
            points_pred_dy = points_pred[:, :, 0::2]
            points_pred_dx = points_pred[:, :, 1::2]
            pts = torch.cat([points_pred_dx, points_pred_dy], dim=2).reshape(-1, 2 * self.num_points)  # pts  torch.size(num_topk, points_num*2) x_first

            pts_pos_center = points[:, :2].repeat(1, self.num_points)                                  # 本尺度预测的真实points append进mlvl_reppoints列表
            pts = pts * self.point_strides[i_lvl] + pts_pos_center

            mlvl_reppoints.append(pts)
        mlvl_bboxes = torch.cat(mlvl_bboxes)                            # mlvl_bboxes list[5*tensor]  -> torch.size(num_topk_total, points_num*2)
        mlvl_reppoints = torch.cat(mlvl_reppoints)                      # mlvl_reppoints list[5*tensor]  -> torch.size(num_topk_total, 8)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)         # 有等比缩放的话就还原到原来的尺度
            mlvl_reppoints /= mlvl_reppoints.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)                            # mlvl_scores list[5*tensor]  -> torch.size(num_topk_total, num_classes)
        if self.use_sigmoid_cls:  # 把背景类别添加到首位
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)      # torch.size(num_topk_total, num_classes) -> torch.size(num_topk_total, num_classes+1)
        if nms:
            det_bboxes, det_labels = multiclass_rnms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, multi_reppoints=mlvl_reppoints)
            return det_bboxes, det_labels      # return det_bboxes:(n, [18, 8, score]), det_labels:(n)
        else:
            return mlvl_reppoints, mlvl_bboxes, mlvl_scores    # return mlvl_reppoints(n, 18), mlvl_bboxes:(n, 8), det_labels:(n, num_classes+1)
            