from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from mmdet.core import rbbox2result, multiclass_rnms
import torch

@DETECTORS.register_module
class OrientedRepPointsDetector(SingleStageDetector):
    """ Oriented RepPoints: Point Set Representation for Aerial Object Detection.
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(OrientedRepPointsDetector,
              self).__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                             pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_rbboxes_ignore=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_rbboxes_ignore=gt_rbboxes_ignore)
        return losses
    
    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img) 
        outs = self.bbox_head(x)  # list[cls_out, pts_out_init, pts_out_refine]  cls_out:list[5 * (b, num_class, f_h, f_w)]
        bbox_inputs = outs + (img_metas, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)  # len=batch_size :list(batch_size * list[det_bboxes, det_labels])  det_bboxes:(n, [18, 8, score]), det_labels:(n)
        bbox_results = [   # list[batch_size * list[num_classes * array(n, [18, 8, score])]]
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)  # return list(num_classes * array)  array.size(n, [18, 8, score])
            for det_bboxes, det_labels in bbox_list  # 遍历每张图像
        ]
        return bbox_results[0]  # 只返回第一张图的检测结果

    def aug_test(self, 
                 imgs, 
                 img_metas, 
                 rescale=False):
        # recompute feats to save memory
        feats = self.extract_feats(imgs)

        aug_reppoints = []
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, rescale, False) # 先不进行rnms
            # (n, 18) (n, 8) (n, [0]+num_classes)
            det_reppoints, det_bboxes, det_scores = self.bbox_head.get_bboxes(*bbox_inputs)[0] # list[mlvl_reppoints, mlvl_bboxes, mlvl_scores] 
            aug_reppoints.append(det_reppoints) # list:[增强的尺度数量 * (n, 18)]
            aug_bboxes.append(det_bboxes)  # list:[增强的尺度数量 * (n, 8)]
            aug_scores.append(det_scores)  # list:[增强的尺度数量 * (n, [0]+num_classes)]

        # merging
        reppoints = torch.cat(aug_reppoints, dim=0) # tensor.size(?, 18)
        bboxes = torch.cat(aug_bboxes, dim=0)       # tensor.size(?, 8)
        scores = torch.cat(aug_scores, dim=0)       # tensor.size(?, [0]+num_classes)

        det_bboxes, det_labels = multiclass_rnms(bboxes, scores,     # (k, [18, 8, score]) and (k, 1)
                                                    self.test_cfg.score_thr, self.test_cfg.nms,
                                                    self.test_cfg.max_per_img, multi_reppoints=reppoints)
                                                    
        bbox_results = rbbox2result(det_bboxes, det_labels,   # list(num_classes * array)  array.size(n, [18, 8, score])
                                   self.bbox_head.num_classes)
        return bbox_results  # 返回一张图的多尺度测试的合并结果
        
        
