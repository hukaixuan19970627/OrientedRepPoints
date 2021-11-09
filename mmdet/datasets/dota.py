from .coco import CocoDataset
from .registry import DATASETS
import numpy as np

@DATASETS.register_module
class DotaDatasetv1(CocoDataset):

    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
         'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor',
         'swimming-pool', 'helicopter')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 Mosaic4=False,
                 Mosaic9=False,
                 Mixup=False):
        """
        DOTA Class

        Args:
            filter_empty_gt (bool): Whether or not filter the empty img and small img.
            Mosaic4(bool): The loading_mode of img, load 4 imgs per iter for mosaic augmentation.
            Mosaic9(bool): The loading_mode of img, load 9 imgs per iter for mosaic augmentation.
            Mixup(bool): The loading_mode of img, load 2 x (1/4/9) imgs per iter for Mixup augmentation.
        """
        
        super(DotaDatasetv1, self).__init__(ann_file,
                                           pipeline,
                                           data_root,
                                           img_prefix,
                                           seg_prefix,
                                           proposal_file,
                                           test_mode,
                                           filter_empty_gt)

        assert (Mosaic4 and Mosaic9) == False
        #assert (Mixup and (Mosaic9 or Mosaic4)) or (Mixup == False)
        if Mosaic4:
            self.Mosaic_mode = 'Mosaic4'
        elif Mosaic9:
            self.Mosaic_mode = 'Mosaic9'
        else:
            self.Mosaic_mode = 'Normal'
        self.Mixup_mode = Mixup

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            # if ann['area'] <= 80: # gt最小外接水平矩形区域像素小于80时
            #     continue

            bbox= ann['bbox']
            
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 8), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 8), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
    
    def prepare_train_img(self, idx):
        """
        Three modes to prepare train_img

        self.Mosaic_mode='Normal' : prepare 1 train_img.
        self.Mosaic_mode='Mosaic4' : prepare 1 train_img.
        self.Mosaic_mode='Mosaic9' : prepare 1 train_img.
        self.Mixup_mode: prepare 2 * Mosaic_mode train_img
        """
        if self.Mosaic_mode == 'Mosaic4':
            num_img = 4
        if self.Mosaic_mode == 'Mosaic9':
            num_img = 9
        if self.Mosaic_mode == 'Normal':
            num_img = 1
        if self.Mixup_mode:
            num_img = num_img * 2
        
        if num_img > 1:
            results_num = []
            for i in range(num_img):
                img_info = self.img_infos[idx]
                ann_info = self.get_ann_info(idx)
                results = dict(img_info=img_info, ann_info=ann_info)

                results['Mosaic_mode'] = self.Mosaic_mode
                results['Mixup_mode'] = True if self.Mixup_mode else False

                if self.proposals is not None:
                    results['proposals'] = self.proposals[idx]
                self.pre_pipeline(results)

                idx = self._rand_another(idx)  # 随机取另一个图像数据的索引
                results_num.append(results)
            return self.pipeline(results_num)
        
        else:
            img_info = self.img_infos[idx]
            ann_info = self.get_ann_info(idx)
            results = dict(img_info=img_info, ann_info=ann_info)

            results['Mosaic_mode'] = self.Mosaic_mode
            results['Mixup_mode'] = False

            if self.proposals is not None:
                results['proposals'] = self.proposals[idx]
            self.pre_pipeline(results)
            return self.pipeline(results)  # 进入compose运行train时的pipeline


@DATASETS.register_module
class DotaDatasetv1_5(DotaDatasetv1):

    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
         'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor',
         'swimming-pool', 'helicopter', 'container-crane')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 Mosaic4=False,
                 Mosaic9=False,
                 Mixup=False):
        
        super(DotaDatasetv1_5, self).__init__(ann_file,
                                           pipeline,
                                           data_root,
                                           img_prefix,
                                           seg_prefix,
                                           proposal_file,
                                           test_mode,
                                           filter_empty_gt,
                                           Mosaic4,
                                           Mosaic9,
                                           Mixup)

@DATASETS.register_module
class DotaDatasetv2(DotaDatasetv1):

    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
         'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor',
         'swimming-pool', 'helicopter', 'container-crane', 'airport', 'helipad')


    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 Mosaic4=False,
                 Mosaic9=False,
                 Mixup=False):
        
        super(DotaDatasetv2, self).__init__(ann_file,
                                           pipeline,
                                           data_root,
                                           img_prefix,
                                           seg_prefix,
                                           proposal_file,
                                           test_mode,
                                           filter_empty_gt,
                                           Mosaic4,
                                           Mosaic9,
                                           Mixup)

@DATASETS.register_module
class DotaDatasetv2_class_under10000(DotaDatasetv1):

    CLASSES = ('baseball-diamond', 'ground-track-field', 'tennis-court', 'basketball-court', 'soccer-ball-field', 'roundabout',
                                'helicopter', 'container-crane', 'airport', 'helipad')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 Mosaic4=False,
                 Mosaic9=False,
                 Mixup=False):
        
        super(DotaDatasetv2_class_under10000, self).__init__(ann_file,
                                           pipeline,
                                           data_root,
                                           img_prefix,
                                           seg_prefix,
                                           proposal_file,
                                           test_mode,
                                           filter_empty_gt,
                                           Mosaic4,
                                           Mosaic9,
                                           Mixup)
                        
@DATASETS.register_module
class DotaDatasetv2_class_above10000(DotaDatasetv1):

    CLASSES = ('plane', 'bridge', 'small-vehicle', 'large-vehicle', 'ship', 'storage-tank', 'harbor', 'swimming-pool')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 Mosaic4=False,
                 Mosaic9=False,
                 Mixup=False):
        
        super(DotaDatasetv2_class_above10000, self).__init__(ann_file,
                                           pipeline,
                                           data_root,
                                           img_prefix,
                                           seg_prefix,
                                           proposal_file,
                                           test_mode,
                                           filter_empty_gt,
                                           Mosaic4,
                                           Mosaic9,
                                           Mixup)
                                           
