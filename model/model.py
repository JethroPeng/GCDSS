import os
import torch
from model import vision_transformer as vit

def load_mask_model(model_name,device):
    model_list = ['sam']
    if model_name not in model_list:
        raise ValueError("Model Error")
    print("load " + model_name) 
    from model.sam.segment_anything  import SamAutomaticMaskGenerator, sam_model_registry
    cwd = os.getcwd()
    sam_path = os.path.join(cwd, 'model_weights/sam_vit_h_4b8939.pth')
    sam = sam_model_registry['vit_h'](checkpoint=sam_path)
    sam.to(device=device)
    for m in sam.parameters():
        m.requires_grad = False
    model = SamAutomaticMaskGenerator(model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100)
    return model


def load_feature_model(model_name,device):
    model_list = ['clip','ovseg','dinov1','dinov2','sam']
    # The model weights below all utilize the best open-source weights known to me.
    if model_name not in model_list:
        raise ValueError("Model Error")
    print("load "+model_name) 
    if model_name == 'dinov1':
        model = vit.__dict__['vit_base']()
        weight = torch.load('model_weights/dino_vitbase16_pretrain_full_checkpoint.pth', map_location='cpu')
        new_state_dict = {}
        for k,v in weight.items():
            if k == "student":
                get_weight = v
                for name,param in get_weight.items():
                    if name.startswith('module.backbone.'):
                        name = name[16:]
                        new_state_dict[name] = param
                    elif name.startswith('module.head.'):
                        continue
                    else:
                        new_state_dict[name] = param
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(msg)
        model.to(device=device)
    elif model_name == 'sam':
        from model.sam.segment_anything  import SamPredictor, sam_model_registry
        cwd = os.getcwd()
        sam_path = os.path.join(cwd, 'model_weights/sam_vit_h_4b8939.pth')
        sam = sam_model_registry['vit_h'](checkpoint=sam_path)
        sam.to(device=device)
        model = SamPredictor(sam) 
    elif model_name == 'clip':
        from model.CLIP import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
    elif model_name == 'ovseg':
        import open_clip
        model,_,_ = open_clip.create_model_and_transforms("ViT-L-14","model_weights/ovseg_clip_l_9a1909.pth") 
        model.to(device=device)
    else:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_lc')
        model.to(device=device)
    if  model_name != 'sam':
        for m in model.parameters():
            m.requires_grad = False
    return model
        
        
