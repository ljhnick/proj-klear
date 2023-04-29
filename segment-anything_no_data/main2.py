from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import clip
from PIL import Image
import os
import time
from sklearn.preprocessing import normalize
from os.path import exists
import torch.nn as nn
import torch.nn.functional as F
from slic_vit import SLICViT
device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_clip_model, preprocess_img = clip.load("ViT-B/32", device=device)
pretrained_clip_model = pretrained_clip_model.to(device)

def generate_clip2d_SAM(image, output_path, flag):
    if not exists(output_path + flag + "_masks.pt"):
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
        torch.save(masks, output_path + flag + "_masks.pt")
    masks = torch.load(output_path + flag + "_masks.pt", map_location = device)
    masks_number = len(masks)
    print(masks_number)
    #Generate clip2d
    image_h = image.shape[0]
    image_w = image.shape[1]
    clip2d = torch.zeros((image_h, image_w, 512)).to(device)
    for i in range(masks_number):
        print(i)
        masked_image = image * np.expand_dims(masks[i]["segmentation"], -1)
        if not exists(output_path + flag + str(i) + "_masked_image.png"):
            masked_image = cv2.cvtColor(np.array(masked_image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path + flag + str(i) + "_masked_image.png", np.array(masked_image))
        masked_image = Image.fromarray(masked_image)
        preprocessed_masked_image = preprocess_img(masked_image).unsqueeze(0).to(device)
        with torch.no_grad():
            masked_image_clip1d = torch.squeeze(pretrained_clip_model.encode_image(preprocessed_masked_image))
        masked_image_clip1d_normalized = masked_image_clip1d.div(masked_image_clip1d.norm(p = 2).expand_as(masked_image_clip1d))
        masked_image_clip1d_final = (clip2d[torch.tensor(masks[i]["segmentation"])] + masked_image_clip1d_normalized)[0]
        masked_image_clip1d_final_normalized = masked_image_clip1d_final.div(masked_image_clip1d_final.norm(p = 2).expand_as(masked_image_clip1d_final))
        clip2d[torch.tensor(masks[i]["segmentation"])] = masked_image_clip1d_final_normalized
    #save clip2d
    np.save(output_path + flag + "_clip2d", clip2d.detach().cpu().numpy())
    cv2.imwrite(output_path + flag + "_clip2d_partial.png", (clip2d[:,:,0:3] * 255).type(dtype=torch.uint8).detach().cpu().numpy())

def generate_clip2d_Sliding_window_Past(image, output_path, flag):
    #get_clipmap()
    window_size = 3
    n_segments = [5]
    patch_size = 32
    target_size = 7
    upsample = 2
    target_size = target_size * upsample
    align_corners = None
    start_block = 0
    #get_mask_features()
    with torch.no_grad():
        h, w = image.shape[:2]
        image = Image.fromarray(image).convert('RGB')
        #image = image.resize((224, 224))
        #get_masks()
        np_image = np.array(image)
        masks = []
        detection_areas = []
        # Do SLIC with different number of segments so that it has a hierarchical scale structure
        # This can average out spurious activations that happens sometimes when the segments are too small
        for n in n_segments:
            # segments_slic = slic(im.astype(
            #     np.float32)/255., n_segments=n, compactness=self.compactness, sigma=self.sigma)
            # print("n:", n)
            # print("segments:",type(segments_slic))
            #seg()
            img = np_image.astype(np.float32)/255.
            seg = pow(2,n)
            # print("shape:", img.shape, type(img))
            w = int(img.shape[0]/seg)
            h = int(img.shape[1]/seg)
            # print("w,h:",w,h)
            window_size = window_size
            r  = int((window_size-1)/2)
            plate = np.zeros((img.shape[0],img.shape[1]))
            areas = []
            n = 0
            for i in range(seg):
                for j in range(seg):
                    if n != 0:
                        mask = n*np.ones((w,h))
                        plate[i*w:(i+1)*w, j*h:(j+1)*h] = mask
                    canvas = np.zeros((img.shape[0],img.shape[1]))
                    for x in range(window_size):
                        for y in range (window_size):
                            canvas[max(0,(i-r+x)*w):min(img.shape[0],(i+1-r+x)*w), max(0,(j-r+y)*h):min(img.shape[1],(j+1-r+y)*h)] = n
                    # print("plate:")
                    # print(plate)
                    # print("canvas:")
                    # print(canvas)
                    areas.append(canvas)
                    n = n + 1
            oct_seg = plate
            for i in np.unique(oct_seg):
                mask = oct_seg == i
                b_mask = areas[int(i)] == i
                # print(mask)
                masks.append(mask)
                detection_areas.append(b_mask)
        masks = np.stack(masks, 0)
        detection_areas = np.stack(detection_areas, 0)
        masks = torch.from_numpy(masks.astype(np.bool)).to(device)
        detection_areas = torch.from_numpy(detection_areas.astype(np.bool)).to(device)
        image = preprocess_img(image).unsqueeze(0).to(device)
        #Reset pretrained_clip_model()
        pretrained_clip_model.visual.conv1.stride = patch_size // upsample
        pretrained_clip_model.visual.conv1.padding = (patch_size - 1) // 2  # TODO: make it more precise
        #upsample_pos_emb()
        emb = pretrained_clip_model.visual.positional_embedding
        # upsample the pretrained embedding for higher resolution
        # emb size NxD
        first = emb[:1, :]
        emb = emb[1:, :]
        N, D = emb.size(0), emb.size(1)
        size = int(np.sqrt(N))
        assert size * size == N
        new_size = size * upsample
        emb = emb.permute(1, 0)
        emb = emb.view(1, D, size, size).contiguous()
        emb = F.upsample(emb, size=new_size, mode='bilinear',
                         align_corners=align_corners)
        emb = emb.view(D, -1).contiguous()
        emb = emb.permute(1, 0)
        emb = torch.cat([first, emb], 0)
        emb = nn.parameter.Parameter(emb.half())
        pretrained_clip_model.visual.positional_embedding = emb
        #model(image, detection_areas)
        vit = pretrained_clip_model.visual
        x = image.type(pretrained_clip_model.dtype)
        x = vit.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + vit.positional_embedding.to(x.dtype)
        x = vit.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # masks size NxHxW
        N = detection_areas.size(0)
        # masks is 1 for the object and 0 for others, need to invert it
        detection_areas = 1 - detection_areas.bool().float()
        # interpolate to target size
        detection_areas = detection_areas.unsqueeze(1).float()
        target_size = (target_size, target_size)
        detection_areas = F.interpolate(detection_areas, size=target_size,
                              mode='nearest', align_corners=None)
        detection_areas = detection_areas.squeeze(1)
        attn_map = detection_areas.view(N, -1)
        attn_map = torch.cat([attn_map, 1-torch.eye(N).to(attn_map.device)], 1)
        attn_map = attn_map.bool().half() * (-100)
        attn_mask = attn_map
        attn_mask = attn_mask.type(pretrained_clip_model.dtype)
        num_masks = attn_mask.size(0)
        for block_idx, resblock in enumerate(vit.transformer.resblocks):
            if block_idx == start_block:
                gv = x[:1]
                gv = gv.repeat(num_masks, 1, 1)  # LND
            if block_idx >= start_block:
                attn = resblock.attn
                source = resblock.ln_1(torch.cat([x[1:], gv], 0))
                gv = gv + attn(
                    source[-num_masks:],
                    source,
                    source,
                    need_weights=False,
                    attn_mask=attn_mask,
                )[0]
                gv = gv + resblock.mlp(resblock.ln_2(gv))
            x = resblock(x)
        gv = gv.permute(1, 0, 2)
        gv = vit.ln_post(gv)
        if vit.proj is not None:
            gv = (gv.view(-1, gv.size(-1)) @
                  vit.proj).view(gv.size(0), gv.size(1), -1)
        image_features = gv
        image_features = torch.reshape(image_features, (image_features.shape[1], image_features.shape[2]))
        print("image_features in clipmap:" , image_features.shape)
        # image_features = torch.permute(image_features,(1, 0))
        image_features = image_features.cpu().float().numpy()
        print("image feature converted to numpy" , image_features.shape)
    masks = masks.cpu().numpy()
    clip_features = image_features
    print("mask shape, feature shape:" , masks.shape, clip_features.shape)
    featuremap = (np.nan + np.zeros((masks.shape[1], masks.shape[2],clip_features.shape[1] ), dtype=np.float32))
    # print("featuremap shape", featuremap.shape)
    print("i goes up to:", len(masks))
    for i in range(len(masks)):
        mask = masks[i]
        # print("mask:",mask.shape)
        features = clip_features[i]
        # print("clip features: ", features.shape)
        # print(featuremap.shape)
        featuremap[mask] = features
    featuremap = np.stack(featuremap, 0)
    # print("heatmap:", featuremap.shape)
    #save clip2d
    featuremap = torch.tensor(featuremap)
    np.save(output_path + flag + "_clip2d", featuremap.detach().cpu().numpy())
    cv2.imwrite(output_path + flag + "_clip2d_partial.png", (featuremap[:,:,0:3] * 255).type(dtype=torch.uint8).detach().cpu().numpy())

def generate_clip2d_Sliding_window(image, output_path, flag):
    args = {
        'model': 'vit32',
        'alpha': 0.75,
        'aggregation': 'mean',
        'n_segments': [5],
        'temperature': 0.02,
        'upsample': 2,
        'start_block': 0,
        'compactness': 50,
        'sigma': 0,
    }
    model = SLICViT(**args).to(device)
    clip2d = torch.tensor(model.get_clipmap(image)) 
    #save clip2d
    np.save(output_path + flag + "_clip2d", clip2d.detach().cpu().numpy())
    cv2.imwrite(output_path + flag + "_clip2d_partial.png", (clip2d[:,:,0:3] * 255).type(dtype=torch.uint8).detach().cpu().numpy())

    


if __name__ == "__main__":
    flag = "lerf_teatime"
    method = "SAM"
    input_path = "data/"
    output_path = "output_data/" + method + "/"
    image = cv2.cvtColor(cv2.imread(input_path + flag + ".png"), cv2.COLOR_BGR2RGB)
    if not exists(output_path + flag + "_clip2d.npy"):
        if method == "SAM":
            #SAM is multi-level by nature (Future work: You can first sort mask by its area, then add overlapped clip with smaller mask area by small factor eg 0.2)
            #How to do multi-layer interpolation
            generate_clip2d_SAM(image, output_path, flag)
        elif method == "Sliding_window":
            generate_clip2d_Sliding_window(image, output_path, flag)
    #一.Test dog (thr > 0.25)
    #1. (435, 180, white dog 0.2645 > dog 0.2636 > white pug 0.2593 > white Shiba Inu 0.2290 > cat 0.2134)
    #2. (211, 713, human legs with blue pants 0.3019 > human legs without blue pants 0.2831 > pants 0.2466 > dog 0.2168)
    #3. (229, 600, silver round bowl 0.2739 > bowl 0.2683)
    #二.Test lerf_teatime (thr > 0.25)
    #Succeed
    #1. red apple
    #2. stuffed bear
    #3. coffee-mug
    #4. black wheel
    #Fail
    #
    clip2d = torch.from_numpy(np.load(output_path + flag + "_clip2d.npy"))
    text = "red apple"
    with torch.no_grad():
        text_clip1d = torch.squeeze(pretrained_clip_model.encode_text(clip.tokenize([text])))
    text_clip1d_normalized = text_clip1d.div(text_clip1d.norm(p=2).expand_as(text_clip1d))
    thr = 0
    saliency2d = torch.zeros((clip2d.shape[0], clip2d.shape[1], 1))
    for i in range(clip2d.shape[0]):
        for j in range(clip2d.shape[1]):
            x_normalized = clip2d[i,j]
            score = torch.dot(x_normalized, text_clip1d_normalized)
            if score > thr:
                saliency2d[i,j] = score
    saliency2d = torch.squeeze(saliency2d)
    saliency2d_h = saliency2d.shape[0]
    saliency2d_w = saliency2d.shape[1]
    saliency2d_reshaped = torch.reshape(saliency2d, (saliency2d_h * saliency2d_w, 1))
    saliency2d_reshaped_normalized = (saliency2d_reshaped - torch.min(saliency2d_reshaped)) / (torch.max(saliency2d_reshaped) - torch.min(saliency2d_reshaped))
    saliency2d_normalized = torch.reshape(saliency2d_reshaped_normalized, (saliency2d_h, saliency2d_w))
    saliency2d_normalized_255 = (saliency2d_normalized * 255).type(torch.uint8)
    cv2.imwrite(output_path + flag + "_" + text + "_" + "saliency2d.png", saliency2d_normalized_255.detach().numpy())

    #___________________
    args = {
        'model': 'vit32',
        'alpha': 0.75,
        'aggregation': 'mean',
        'n_segments': [5],
        'temperature': 0.02,
        'upsample': 2,
        'start_block': 0,
        'compactness': 50,
        'sigma': 0,
    }
    model = SLICViT(**args).to(device)
    query_map = model.verify(clip2d, text, output_path)
    query_map_scores = np.squeeze(query_map)
    print(query_map_scores.shape)
    print(np.min(query_map_scores))
    query_map_remapped = (query_map_scores - np.min(query_map_scores)) / (np.max(query_map_scores) - np.min(query_map_scores))
    np.save(output_path + flag + "_" + text + "_" + "saliency2d_color", query_map_remapped)
    query_map = query_map.reshape(query_map.shape[0], query_map.shape[1])
    indices = np.where(query_map_remapped >= 1-0.2)
    # print(indices)
    x_y_coords =list(zip(indices[0], indices[1]))
    # print(x_y_coords)
    MAXMAP = np.zeros(query_map.shape)
    for index in x_y_coords:
        MAXMAP[index]=1
    plt.imshow(query_map)
    # plt.imshow(query_map_3d)
    plt.imsave(output_path + flag + "_" + text + "_" + "saliency2d_color.png", query_map)
    plt.imshow(MAXMAP)
    # plt.imshow(query_map_3d)
    plt.imsave(output_path + flag + "_" + text + "_" + "saliency2d_Max_color.png", MAXMAP)


    