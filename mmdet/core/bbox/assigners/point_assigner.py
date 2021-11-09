import torch

from .assign_result import AssignResult
from .base_assigner import BaseAssigner


class PointAssigner(BaseAssigner):
    """Assign a corresponding gt rbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    """

    def __init__(self, scale=4, pos_num=3):
        self.scale = scale
        self.pos_num = pos_num

    def assign(self, points, gt_rbboxes, gt_rbboxes_ignore=None, gt_labels=None):

        """Assign gt to points.

                This method assign a gt bbox to every points set, each points set
                will be assigned with  0, or a positive number.
                0 means negative sample, positive number is the index (1-based) of
                assigned gt.
                The assignment is done in following steps, the order matters.

                1. assign every points to 0
                2. A point is assigned to some gt bbox if
                    (i) the point is within the k closest points to the gt bbox
                    (ii) the distance between this point and the gt is smaller than
                        other gt bboxes

                Args:
                    points (Tensor): points to be assigned, shape(n, 3) while last
                        dimension stands for (x, y, stride).
                    gt_rboxes (Tensor): Groundtruth boxes, shape (k, 8).
                    gt_rboxes_ignore (Tensor, optional): Ground truth bboxes that are
                        labelled as `ignored`, e.g., crowd boxes in COCO.
                        NOTE: currently unused.
                    gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

                Returns:
                    :obj:`AssignResult`: The assign result.
                """

        num_points = points.shape[0]
        num_gts = gt_rbboxes.shape[0]

        if num_gts == 0 or num_points == 0:
            # If no truth assign everything to the background
            assigned_gt_inds = points.new_full((num_points,),
                                               0,
                                               dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_zeros((num_points,),
                                                   dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(
            points_stride).int()  # [3...,4...,5...,6...,7...]
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        assert gt_rbboxes.size(1) == 8, 'gt_rbboxes should be (N * 8)'

        # gt_rbboxes convert to gt_bbox
        gt_xs, gt_ys = gt_rbboxes[:, 0::2], gt_rbboxes[:, 1::2]
        gt_xmin, _ = gt_xs.min(1)
        gt_ymin, _ = gt_ys.min(1)
        gt_xmax, _ = gt_xs.max(1)
        gt_ymax, _ = gt_ys.max(1)
        gt_bboxes = torch.cat([gt_xmin[:, None], gt_ymin[:, None],
                               gt_xmax[:, None], gt_ymax[:, None]], dim=1)

        # assign gt rbox   分配gtbox给对应的point尺度
        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2

        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
        scale = self.scale
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                          torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        # stores the assigned gt index of each point
        assigned_gt_inds = points.new_zeros((num_points,), dtype=torch.long) # 用于存储每个point是否为正样本 每个点先标记为背景 赋值0 值为多少则表示与第几个gtbox匹配
        # stores the assigned gt dist (to this point) of each point
        assigned_gt_dist = points.new_full((num_points,), float('inf')) # 用于存储每个点离gtbox的最小距离 每个点先标记为背景 因此赋值inf无穷大
        points_range = torch.arange(points.shape[0])

        for idx in range(num_gts):             # 遍历每个gtbox
            gt_lvl = gt_bboxes_lvl[idx]        # 获取当前gtbox被分配到的尺度
            # get the index of points in this level
            lvl_idx = gt_lvl == points_lvl        # gt_lvl与point_lvl做比较，尺度相同的索引置True
            points_index = points_range[lvl_idx]  # 取出与gtbox同尺度的points的索引
            # get the points in this level
            lvl_points = points_xy[lvl_idx, :]    # 取出与gtbox同尺度的points  torch.size(points_samestride_num, 2) last dimension stands for[x, y]
            # get the center point of gt
            gt_point = gt_bboxes_xy[[idx], :]     # 取出当前gtbox的中心点坐标 torch.size(1, 2)  last dimension stands for[x, y]
            # get width and height of gt
            gt_wh = gt_bboxes_wh[[idx], :]        # 取出当前gtbox的长宽信息 torch.size(1, 2) last dimension stands for[x, y]
            # compute the distance between gt center and
            #   all points in this level
            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)  # 当前尺度的所有point与该gtbox 中心点的距离  torch.size(points_samestride_num)
            # find the nearest k points to gt center in this level
            min_dist, min_dist_index = torch.topk(                          
                points_gt_dist, self.pos_num, largest=False)                # 距离topk排序，获得前pos_num个最近的point的距离值与对应的索引
            # the index of nearest k points to gt center in this level
            min_dist_points_index = points_index[min_dist_index]            # 在同尺度的points索引中 ，再取索引[pos_num]，获得实际的topk points的索引

            # The less_than_recorded_index stores the index
            #   of min_dist that is less then the assigned_gt_dist. Where
            #   assigned_gt_dist stores the dist from previous assigned gt
            #   (if exist) to each point. 
            less_than_recorded_index = min_dist < assigned_gt_dist[           # 最新的gtbox与points之间的距离 是否 比之前的gtbox还小
                min_dist_points_index]
            # The min_dist_points_index stores the index of points satisfy:
            #   (1) it is k nearest to current gt center in this level.
            #   (2) it is closer to current gt center than other gt center.
            min_dist_points_index = min_dist_points_index[                    # 取出更近的gtbox的points实际索引
                less_than_recorded_index]
            # assign the result
            assigned_gt_inds[min_dist_points_index] = idx + 1                 # 该点可以与该gtbox匹配，值设置为框的索引+1
            assigned_gt_dist[min_dist_points_index] = min_dist[               # 该点与gtbox的最小距离更新
                less_than_recorded_index]

        if gt_labels is not None:                                             
            assigned_labels = assigned_gt_inds.new_zeros((num_points,))       # 每个point先全部标记为背景0 torch.size(num_points)
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()          # 找出本次gtbox标记的正样本points真实索引 torch.size(topk * num_gtbox )
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[                        # 把对应框的labels赋值给对应的points
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)  # num_gts:gt数量, assigned_gt_inds：每个point匹配的第几个gt， assigned_labels：每个point的label
                                                                      # torch.size(num_gts),   torch.size(num_points,)            torch.size(num_points,)
