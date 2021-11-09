import torch

from mmdet.ops.nms import nms_wrapper


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   multi_reppoints=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    
    if multi_reppoints is not None:
        reppoints = multi_reppoints[:, None].expand(-1, num_classes, multi_reppoints.size(-1))
        
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    
    if multi_reppoints is not None:
        reppoints = reppoints[valid_mask] 
        
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        if multi_reppoints is None:
            bboxes = multi_bboxes.new_zeros((0, 5))
        else:
            bboxes = multi_bboxes.new_zeros((0, reppoints.size(-1) + 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    dets, keep = nms_op(
        torch.cat([bboxes_for_nms, scores[:, None]], 1), **nms_cfg_)
    bboxes = bboxes[keep]
    if multi_reppoints is not None:
        reppoints = reppoints[keep]
        bboxes = torch.cat([reppoints, bboxes], dim=1)
    scores = dets[:, -1]  # soft_nms will modify scores
    labels = labels[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]

    return torch.cat([bboxes, scores[:, None]], 1), labels

def multiclass_rnms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   multi_reppoints=None):
    """NMS for multi-class rbboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*8) or (n, 8)
        multi_scores (Tensor): shape (n, #class + 1), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS
        multi_reppoints (Tensor): shape (n, num_points*2)
    Returns:
        tuple: (bboxes, labels), tensors of shape (k, [18, 8, score]) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 8:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 8)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 8)          # num_class通道处的数据全部相等
        
    if multi_reppoints is not None:
        reppoints = multi_reppoints[:, None].expand(-1, num_classes, multi_reppoints.size(-1))
        
    scores = multi_scores[:, 1:]               # 过滤背景的scores  shape(num, num_classes)

    # filter out boxes with low scores         # conf_thresh filter
    valid_mask = scores > score_thr            # 
    bboxes = bboxes[valid_mask]                # tesnor torch.size(num_conf_filter, 8)  scores>阈值的类别全部保留
    
    if multi_reppoints is not None:
        reppoints = reppoints[valid_mask]      # tesnor torch.size(num_conf_filter, 2*num_points)  scores>阈值的类别全部保留
        
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]                # tensor torch.size(num_conf_filter)
    labels = valid_mask.nonzero()[:, 1]        # 保留类别索引id  torch.size(num_conf_filter)
    
    if bboxes.numel() == 0:
        if multi_reppoints is None:
            bboxes = multi_bboxes.new_zeros((0, 9))
        else:
            bboxes = multi_bboxes.new_zeros((0, reppoints.size(-1) + 9))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]              # 同类别仅与同类别box进行nms
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'rnms')
    nms_op = getattr(nms_wrapper, nms_type)
    
    dets, keep = nms_op(                                    # bboxes_for_nms torch.size(n, 8)  scores torch.size(n) -> torch.size(n,)
        torch.cat([bboxes_for_nms, scores[:, None]], 1), **nms_cfg_)  # dets torch.size(n_nms, 9) 9: [bbox, score] ;  keep torch.size(n_nms)
    
    bboxes = bboxes[keep]                                   # bboxes : torch.szie(n_nms, 8)
    # print('bboxes_nms', bboxes)
    if multi_reppoints is not None:
        reppoints = reppoints[keep]                         # reppoints: torch.size(n_nms, points_num*2)  x_first
        bboxes = torch.cat([reppoints, bboxes], dim=1)      # bboxes: torch.size(n_nms, [points_num*2, poly]) x_first
        # print('bboxes_reppoints', bboxes)
    scores = dets[:, -1]  # soft_nms will modify scores     # scores: torch.size(n_nms) 分类scores
    labels = labels[keep]                                   # labels: torch.size(n_nms) 类别索引id

    if keep.size(0) > max_num:                              # 若nms后的数据依旧太多，则依据score从大到小排序，取前max_num
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]

    return torch.cat([bboxes, scores[:, None]], 1), labels  # return pred:(n, [18, 8, score]), labels:(n)
