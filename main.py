import os
import random
import argparse
import numpy as np
import torch
import torch.cuda
from torch.utils.data.dataloader import DataLoader 

from datasets import cityscapes
from utils import *
from funcs import *
from model.model import * 
from util.mask import *


import warnings
warnings.filterwarnings('ignore')

def get_argparser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--data_root', type=str, default='./datasets',
                        help='The root path of datasets')                       
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'],
                        help='Name of dataset to use')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for label')        
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers')                       
    parser.add_argument('--unlabel_batch_size', type=int, default=1,
                        help='Batch size for unlabel')                        
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='ID of GPU to use')                        
    parser.add_argument('--random_seed', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('--label_data_path', type=str, default=None,
                        help='Path of label data list file')                        
    parser.add_argument('--unlabel_data_path', type=str, default=None,  
                        help='Path of unlabel data list file')
    parser.add_argument('--novel_class_list', type=str, default="3,15,16,17",
                        help='ID of novel classes')
    parser.add_argument('--save_path', type=str, default='./result',
                        help='Path to save files')                        
    parser.add_argument('--label_feature_name', type=str, default='label_mask_feature',
                        help='Name of label mask feature file')                        
    parser.add_argument('--unlabel_feature_name', type=str, default='unlabel_mask_feature',
                        help='Name of unlabel mask feature file')                                           
    parser.add_argument('--dis_name', type=str, default="dis.npy",
                        help='File name of distance matrix')                                                   
    parser.add_argument('--pred_name', type=str, default="pred.npy",
                        help='File name of predicted results')      
    parser.add_argument('--ground_truth_name', type=str, default="gt.npy", 
                        help='File name of ground truth')                                            
    parser.add_argument('--mask_model', type=str, default='sam',
                        choices=['sam'],
                        help='Name of model to generate mask')                  
    parser.add_argument('--model', type=str, default='dinov2',
                        choices=['clip','ovseg','dinov1','dinov2','sam'],
                        help='Name of model to extract feature')                                                                 
    parser.add_argument('--theta', type=int, default=200,
                        help='theta for small masks')  
    parser.add_argument('--bbox_scale', type=float, default=0,
                        help='bbox_scale')
    parser.add_argument('--topk', type=float, default=10,
                        help='topk for dis')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='threshold')    
    parser.add_argument('--max_iterations', type=int, default=10,
                        help='max_iterations')  
    parser.add_argument('--pred_new', type=int, default=10,
                        help='pred_new')                    
    return parser

def get_dataset(opts, 
                novel_class_list,
                debug):
    ''' 
    Dataset
    '''
    train_dst = None
    test_dst  = None
    if opts.dataset == 'cityscapes':
       label_dst = cityscapes.Cityscapes_GCD(root=opts.data_root,
                                  image_set='label',  
                                  split_path=opts.label_data_path,
                                  novel_class_list=novel_class_list,
                                  debug=debug)
       unlabel_dst  = cityscapes.Cityscapes_GCD(root=opts.data_root,
                                  image_set='unlabel',
                                  split_path=opts.unlabel_data_path,
                                  novel_class_list=novel_class_list,
                                  debug=debug)
    else:
        raise ValueError('Intestid dataset')
    return label_dst, unlabel_dst

if __name__ == '__main__':
    # init
    opts = get_argparser().parse_args()
    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: %s' % device)
    # seed 
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    novel_class_list = [int(x) for x in opts.novel_class_list.split(",")]
    print('novel class list: ', novel_class_list)
    
    label_feature_path    = os.path.join(opts.save_path, opts.label_feature_name + '_' + opts.model + '_' + str(opts.bbox_scale) + '.txt')
    unlabel_feature_path  = os.path.join(opts.save_path, opts.unlabel_feature_name + '_' + opts.model + '_' + str(opts.bbox_scale) + '.txt')
    

    # dataset
    # One crucial aspect to emphasize is that if you have applied 'shuffle' to the data, 
    # there might be issues when you migrate intermediate data to another server.
    label_dst, unlabel_dst = get_dataset(opts=opts,
                                      novel_class_list=novel_class_list, 
                                      debug=False)
    label_loader = DataLoader(dataset=label_dst, 
                              batch_size=opts.batch_size, 
                              shuffle=True, 
                              num_workers=opts.num_workers)
    unlabel_loader  = DataLoader(dataset=unlabel_dst, 
                              batch_size=opts.unlabel_batch_size, 
                              shuffle=True, 
                              num_workers=opts.num_workers)
    print('Dataset: %s, Label set: %d, Unlabel set: %d' %  (opts.dataset, len(label_loader), len(unlabel_loader)))
    
    
    # Since the overall runtime of the code is quite long
    # the code supports running in stages. Simply select the stage you wish to run.
       
    # begin
    run_mask_generate = True
    run_mask_label_generate = True
    run_mask_to_feature = True
    
    # advanced method
    run_get_dis_array = True
    run_filter_sample = True
    
    
    # baseline
    run_feature_clustering = False
    
    # result
    run_compute_miou = True    
        
    if run_mask_generate:
        mask_generate(label_loader=label_loader, 
                      unlabel_loader=unlabel_loader, 
                      device=device, 
                      model=opts.mask_model, 
                      theta=opts.theta)
                      
    if run_mask_label_generate:
        mask_label_generate(label_loader=label_loader, 
                            unlabel_loader=unlabel_loader, 
                            device=device)
                            
    if run_mask_to_feature:
        mask_to_feature(label_loader=label_loader, 
                        unlabel_loader=unlabel_loader, 
                        device=device, 
                        model_name=opts.model, 
                        label_feature_path  = label_feature_path, 
                        unlabel_feature_path= unlabel_feature_path, 
                        bbox_scale=opts.bbox_scale)
    print(label_feature_path)
    label_data  , label_label_full  , label_label_max  , label_label_area   = read_feature_file(data_loader=label_loader, 
                                                                                                feature_path=label_feature_path,
                                                                                                is_label=True)
    unlabel_data, unlabel_label_full, unlabel_label_max, unlabel_label_area = read_feature_file(data_loader=unlabel_loader, 
                                                                                                feature_path=unlabel_feature_path,
                                                                                                is_label=False)
    if run_get_dis_array:
        get_dis_array(label_loader=label_loader, 
                      unlabel_loader=unlabel_loader, 
                      device=device, 
                      K=opts.topk+1,
                      save_path=opts.save_path, 
                      dis_name=opts.dis_name, 
                      ground_truth_name=opts.ground_truth_name, 
                      label_data=label_data, 
                      label_label_full=label_label_full, 
                      label_label_max=label_label_max, 
                      label_label_area=label_label_area, 
                      unlabel_data=unlabel_data, 
                      unlabel_label_full=unlabel_label_full, 
                      unlabel_label_max=unlabel_label_max, 
                      unlabel_label_area=unlabel_label_area)
    
    if run_filter_sample:
        filter_sample(label_loader=label_loader, 
                      unlabel_loader=unlabel_loader, 
                      device=device, 
                      k=opts.topk,
                      save_path=opts.save_path, 
                      dis_name=opts.dis_name, 
                      ground_truth_name=opts.ground_truth_name, 
                      pred_name=opts.pred_name, 
                      pred_new=opts.pred_new, 
                      threshold=opts.threshold, 
                      max_iterations=opts.max_iterations, 
                      label_data=label_data, 
                      label_label_full=label_label_full, 
                      label_label_max=label_label_max, 
                      label_label_area=label_label_area, 
                      unlabel_data=unlabel_data, 
                      unlabel_label_full=unlabel_label_full, 
                      unlabel_label_max=unlabel_label_max, 
                      unlabel_label_area=unlabel_label_area)
                          
    if run_feature_clustering:
        feature_clustering(label_loader=label_loader, 
                           unlabel_loader=unlabel_loader, 
                           device=device, 
                           save_path=opts.save_path,
                           feature_path=unlabel_feature_path,
                           pred_name=opts.pred_name, 
                           pred_new=opts.pred_new, 
                           label_data=label_data, 
                           label_label_full=label_label_full, 
                           label_label_max=label_label_max, 
                           label_label_area=label_label_area, 
                           unlabel_data=unlabel_data, 
                           unlabel_label_full=unlabel_label_full, 
                           unlabel_label_max=unlabel_label_max, 
                           unlabel_label_area=unlabel_label_area)
                           
    if run_compute_miou:
        compute_miou(label_loader=label_loader, 
                     unlabel_loader=unlabel_loader, 
                     device=device, 
                     save_path=opts.save_path,
                     feature_path=unlabel_feature_path,
                     pred_name=opts.pred_name, 
                     pred_new=opts.pred_new, 
                     label_data=label_data, 
                     label_label_full=label_label_full, 
                     label_label_max=label_label_max, 
                     label_label_area=label_label_area, 
                     unlabel_data=unlabel_data, 
                     unlabel_label_full=unlabel_label_full, 
                     unlabel_label_max=unlabel_label_max, 
                     unlabel_label_area=unlabel_label_area)
    