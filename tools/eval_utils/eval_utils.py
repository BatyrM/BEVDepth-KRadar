import pickle
import time

import numpy as np
import torch
# from tqdm import tqdm
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils.visualize_utils import vis_in_open3d, func_show_lidar_point_cloud

import os

import pcdet.datasets.kradar.kitti_eval.kitti_common as kitti
from pcdet.datasets.kradar.kitti_eval.eval import get_official_eval_result

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    gt_annos = []
    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

def eval_map_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    all_preds_dict = []
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            preds_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'
        all_preds_dict.extend(preds_dict)

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        all_preds_dict = common_utils.merge_results_dist(all_preds_dict, len(dataset), tmpdir=result_dir / 'tmpdir')
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    metric = dataset.evaluation_map_segmentation(
        all_preds_dict
    )
    print(metric)
    logger.info('****************Evaluation done.*****************')
    return metric

def dict_datum_to_kitti(dict_item, pred_dict):
    '''
    * Assuming batch size as 1
    '''
    list_kitti_pred = []
    list_kitti_gt = []
    dict_val_keyword = {'Sedan' : 'sed', 'Bus or Truck' : 'bus'} # without empty space for cls name
    class_names = ['Sedan', 'Bus or Truck']
    dict_cls_id_to_name = dict()
    for idx_cls, cls_name in enumerate(class_names):
        dict_cls_id_to_name[(idx_cls+1)] = cls_name
    # considering gt
    header_gt = '0.00 0 0 50 50 150 150'
    for idx_gt, label in enumerate(dict_item['meta'][0]['label']): # Assuming batch size as 1
        cls_name, (xc, yc, zc, rz, xl, yl, zl), _, _ = label
        xc, yc, zc, rz, xl, yl, zl = np.round(xc, 2), np.round(yc, 2), np.round(zc, 2), np.round(rz, 2), np.round(xl, 2), np.round(yl, 2), np.round(zl, 2),
        cls_val_keyword = dict_val_keyword[cls_name]
        # print(cls_val_keyword)
        box_centers = str(yc) + ' ' + str(zc) + ' ' + str(xc) # xcam, ycam, zcam
        box_dim = str(zl) + ' ' + str(yl) + ' ' + str(xl) # height(ycaml), width(xcaml), length(zcaml)
        str_rot = str(rz)

        kitti_gt = cls_val_keyword + ' ' + header_gt  + ' ' + box_dim  + ' ' + box_centers + ' ' + str_rot
        list_kitti_gt.append(kitti_gt)
    
    if pred_dict['pp_num_bbox'] == 0: # empty prediction (should consider for lpc)
        # KITTI Example: Car -1 -1 -4.2780 668.7884 173.1068 727.8801 198.9699 1.4607 1.7795 4.5159 5.3105 1.4764 43.1853 -4.1569 0.9903
        kitti_dummy = 'dummy -1 -1 0 0 0 0 0 0 0 0 0 0 0 0 0'
        list_kitti_pred.append(kitti_dummy)
    else:
        list_pp_cls = pred_dict['pp_cls']
        header_pred = '-1 -1 0 50 50 150 150'
        for idx_pred, pred_box in enumerate(pred_dict['pp_bbox']):
            score, xc, yc, zc, xl, yl, zl, rot = pred_box
            cls_id = list_pp_cls[idx_pred]
            cls_name = dict_cls_id_to_name[cls_id]
            cls_val_keyword = dict_val_keyword[cls_name]
            
            box_centers = str(yc) + ' ' + str(zc) + ' ' + str(xc)
            box_dim = str(zl) + ' ' + str(yl) + ' ' + str(xl)
            str_rot = str(rot)
            str_score = str(score)
            kitti_pred = cls_val_keyword + ' ' + header_pred  + ' ' + box_dim + ' ' + box_centers + ' ' + str_rot + ' ' + str_score
            list_kitti_pred.append(kitti_pred)

    # pp: post-processing
    dict_desc = pred_dict['pp_desc']
    capture_time = dict_desc['capture_time']
    road_type = dict_desc['road_type']
    climate = dict_desc['climate']

    dict_item['kitti_pred'] = list_kitti_pred
    dict_item['kitti_gt'] = list_kitti_gt
    dict_item['kitti_desc'] = f'{capture_time}\n{road_type}\n{climate}'
    
    return dict_item

def read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def get_list_val_care_idx(class_list = ['Sedan', 'Bus or Truck']):
    val_keyword = {
        'Sedan': 'sed',
        'Bus or Truck': 'bus',
        'Motorcycle': 'mot',
        'Bicycle': 'bic',
        'Bicycle Group': 'big',
        'Pedestrian': 'ped',
        'Pedestrian Group': 'peg'
    } # for kitti_eval
    list_val_keyword_keys = list(val_keyword.keys()) # same order as VAL.CLASS_VAL_KEYWORD.keys()
    list_val_care_idx = []
    for cls_name in class_list:
        idx_val_cls = list_val_keyword_keys.index(cls_name)
        list_val_care_idx.append(idx_val_cls)
    
    return list_val_care_idx

def eval_one_epoch2(cfg, args, model, dataloader, epoch_id, logger, batch_size, dist_test=False, result_dir=None, list_conf_thr = [0.3], is_subset=False):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataloader.dataset
    class_names = dataset.class_names

    model.eval()

    tqdm_bar = tqdm.tqdm(total=len(dataloader), desc='* Test (Total): ')

    if epoch_id is None:
        dir_epoch = 'none'
    else:
        dir_epoch = f'epoch_{epoch_id}_subset' if is_subset else f'epoch_{epoch_id}_total'

    path_dir = os.path.join(result_dir, 'test_kitti', dir_epoch)

    for conf_thr in list_conf_thr:
        os.makedirs(os.path.join(path_dir, f'{conf_thr}'), exist_ok=True)
        with open(path_dir + f'/{conf_thr}/' + 'val.txt', 'w') as f:
            f.write('')
        f.close()

    for idx_datum, dict_datum in enumerate(dataloader):
        
        load_data_to_gpu(dict_datum)

        with torch.no_grad():
            dict_out, ret_dict = model(dict_datum)
        
        idx_name = str(idx_datum).zfill(6)

        for conf_thr in list_conf_thr:
            preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
            labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
            desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
            list_dir = [preds_dir, labels_dir, desc_dir]
            split_path = path_dir + f'/{conf_thr}/' + 'val.txt'
            for temp_dir in list_dir:
                os.makedirs(temp_dir, exist_ok=True)

            pred_dicts = dict_out[0]
            pred_boxes = pred_dicts['pred_boxes'].detach().cpu().numpy()
            pred_scores = pred_dicts['pred_scores'].detach().cpu().numpy()
            pred_labels = pred_dicts['pred_labels'].detach().cpu().numpy()
            list_pp_bbox = []
            list_pp_cls = []

            for idx_pred in range(len(pred_labels)):
                x, y, z, l, w, h, th = pred_boxes[idx_pred]
                score = pred_scores[idx_pred]
                
                if score > conf_thr:
                    cls_idx = int(np.round(pred_labels[idx_pred]))
                    cls_name = class_names[cls_idx-1]
                    list_pp_bbox.append([score, x, y, z, l, w, h, th])
                    list_pp_cls.append(cls_idx)
                else:
                    continue
            pp_num_bbox = len(list_pp_cls)
            dict_out_current = dict_out[0]
            dict_out_current.update({
                'pp_bbox': list_pp_bbox,
                'pp_cls': list_pp_cls,
                'pp_num_bbox': pp_num_bbox,
                'pp_desc': {'climate' : 'no description', 'capture_time' : 'unknown', 'road_type' : 'unknown'}
            })

            dict_out = dict_datum_to_kitti(dict_datum, dict_out[0])
            
            if len(dict_out['kitti_gt']) == 0: # no eval for empty obj label
                    pass
            else:
                ### Gt ###
                for idx_label, label in enumerate(dict_out['kitti_gt']):
                    open_mode = 'w' if idx_label == 0 else 'a'
                    with open(labels_dir + '/' + idx_name + '.txt', open_mode) as f:
                        f.write(label+'\n')
                ### Gt ###

                ### Process description ###
                with open(desc_dir + '/' + idx_name + '.txt', 'w') as f:
                    f.write(dict_out['kitti_desc'])
                ### Process description ###

                ### Pred: do not care len 0 with if else: already care as dummy ###
                for idx_pred, pred in enumerate(dict_out['kitti_pred']):
                    open_mode = 'w' if idx_pred == 0 else 'a'
                    with open(preds_dir + '/' + idx_name + '.txt', open_mode) as f:
                        f.write(pred+'\n')
                ### Pred: do not care len 0 with if else: already care as dummy ###

                str_log = idx_name + '\n'
                with open(split_path, 'a') as f:
                    f.write(str_log)
        tqdm_bar.update(1)
    tqdm_bar.close()

    ### Validate per conf ###
    metrics_dict = dict()
    for conf_thr in list_conf_thr:
        preds_dir = os.path.join(path_dir, f'{conf_thr}', 'pred')
        labels_dir = os.path.join(path_dir, f'{conf_thr}', 'gt')
        desc_dir = os.path.join(path_dir, f'{conf_thr}', 'desc')
        split_path = path_dir + f'/{conf_thr}/' + 'val.txt'

        dt_annos = kitti.get_label_annos(preds_dir)
        val_ids = read_imageset_file(split_path)
        gt_annos = kitti.get_label_annos(labels_dir, val_ids)
        
        # print("GT ANNOS:")
        # print(len(gt_annos))
        # print(gt_annos[0])
        # print("PRED_ANNOS:")
        # print(len(dt_annos))
        # print(dt_annos[0])

        list_val_care_idx = get_list_val_care_idx()

        metrics_dict[conf_thr] = dict()
        for idx_cls_val in list_val_care_idx:
            dict_metrics, result_log = get_official_eval_result(gt_annos, dt_annos, idx_cls_val, is_return_with_dict=True)
            print(f'-----conf{conf_thr}-----')
            # print(result_log)
            with open(os.path.join(path_dir, f'{conf_thr}', 'mAP_bs{}.txt'.format(batch_size)), "a+") as text_file:
                text_file.write(result_log + "\n")
            print(dict_metrics)
            metrics_dict[conf_thr][idx_cls_val] = dict_metrics
    return metrics_dict
    ### Validate per conf ###    

            


if __name__ == '__main__':
    pass
