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
from torch.nn.functional import normalize
from collections import OrderedDict
from scipy.ndimage.filters import gaussian_filter
device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_clip_model, preprocess_img = clip.load("ViT-B/32", device=device)
pretrained_clip_model = pretrained_clip_model.to(device)

def generate_clip2d_SAM_Replacement(flag, image, output_path):
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
        if not exists(output_path + flag + "_masked_images"):
            os.mkdir(output_path + flag + "_masked_images")
        if not exists(output_path + flag + "_masked_images/" + flag + "_masked_image" + "_" + str(i) + ".png"):
            masked_image = cv2.cvtColor(np.array(masked_image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path + flag + "_masked_images/" + flag + "_masked_image" + "_" + str(i) + ".png", np.array(masked_image))
        masked_image = Image.fromarray(masked_image)
        preprocessed_masked_image = preprocess_img(masked_image).unsqueeze(0).to(device)
        with torch.no_grad():
            masked_image_clip1d = torch.squeeze(pretrained_clip_model.encode_image(preprocessed_masked_image))
        masked_image_clip1d_normalized = masked_image_clip1d.div(masked_image_clip1d.norm(p = 2).expand_as(masked_image_clip1d))
        #Replacement
        clip2d[torch.tensor(masks[i]["segmentation"])] = masked_image_clip1d_normalized
    #save clip2d
    np.save(output_path + flag + "_clip2d", clip2d.detach().cpu().numpy())
    cv2.imwrite(output_path + flag + "_clip2d_partial.png", (clip2d[:,:,0:3] * 255).type(dtype=torch.uint8).detach().cpu().numpy())

def generate_clip2d_SAM_Interpolation(flag, image, output_path):
    w1 = 0.9
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
        if not exists(output_path + flag + "_masked_images"):
            os.mkdir(output_path + flag + "_masked_images")
        if not exists(output_path + flag + "_masked_images/" + flag + "_masked_image" + "_" + str(i) + ".png"):
            masked_image = cv2.cvtColor(np.array(masked_image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path + flag + "_masked_images/" + flag + "_masked_image" + "_" + str(i) + ".png", np.array(masked_image))
        masked_image = Image.fromarray(masked_image)
        preprocessed_masked_image = preprocess_img(masked_image).unsqueeze(0).to(device)
        with torch.no_grad():
            masked_image_clip1d = torch.squeeze(pretrained_clip_model.encode_image(preprocessed_masked_image))
        masked_image_clip1d_normalized = masked_image_clip1d.div(masked_image_clip1d.norm(p = 2).expand_as(masked_image_clip1d))
        #Interpolation
        if torch.count_nonzero(clip2d[torch.tensor(masks[i]["segmentation"])]) == 0:
            clip2d[torch.tensor(masks[i]["segmentation"])] = masked_image_clip1d_normalized
        else: #同一个mask下，可以有一半0，一半非0
            clip2d[torch.tensor(masks[i]["segmentation"])] = normalize((w1 * clip2d[torch.tensor(masks[i]["segmentation"])] + (1-w1) * masked_image_clip1d_normalized), p = 2.0, dim = -1)
    #save clip2d
    np.save(output_path + flag + "_clip2d", clip2d.detach().cpu().numpy())
    cv2.imwrite(output_path + flag + "_clip2d_partial.png", (clip2d[:,:,0:3] * 255).type(dtype=torch.uint8).detach().cpu().numpy())

def generate_clip2d_SAM_SortMask_Interpolation(flag, image, output_path):
    w1 = 0.5
    if not exists(output_path + flag + "_masks.pt"):
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
        torch.save(masks, output_path + flag + "_masks.pt")
    masks = torch.load(output_path + flag + "_masks.pt", map_location = device)
    #Sort mask
    masks = sorted(masks, key = lambda x : x["area"], reverse=True)
    masks_number = len(masks)
    print(masks_number)
    #Generate clip2d
    image_h = image.shape[0]
    image_w = image.shape[1]
    clip2d = torch.zeros((image_h, image_w, 512)).to(device)
    for i in range(masks_number):
        print(i)
        masked_image = image * np.expand_dims(masks[i]["segmentation"], -1)
        if not exists(output_path + flag + "_masked_images"):
            os.mkdir(output_path + flag + "_masked_images")
        if not exists(output_path + flag + "_masked_images/" + flag + "_masked_image" + "_" + str(i) + ".png"):
            masked_image = cv2.cvtColor(np.array(masked_image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path + flag + "_masked_images/" + flag + "_masked_image" + "_" + str(i) + ".png", np.array(masked_image))
        masked_image = Image.fromarray(masked_image)
        preprocessed_masked_image = preprocess_img(masked_image).unsqueeze(0).to(device)
        with torch.no_grad():
            masked_image_clip1d = torch.squeeze(pretrained_clip_model.encode_image(preprocessed_masked_image))
        masked_image_clip1d_normalized = masked_image_clip1d.div(masked_image_clip1d.norm(p = 2).expand_as(masked_image_clip1d))
        #Interpolation
        if torch.count_nonzero(clip2d[torch.tensor(masks[i]["segmentation"])]) == 0:
            clip2d[torch.tensor(masks[i]["segmentation"])] = masked_image_clip1d_normalized
        else: #同一个mask下，可以有一半0，一半非0
            clip2d[torch.tensor(masks[i]["segmentation"])] = normalize((w1 * clip2d[torch.tensor(masks[i]["segmentation"])] + (1-w1) * masked_image_clip1d_normalized), p = 2.0, dim = -1)
    #save clip2d
    np.save(output_path + flag + "_clip2d", clip2d.detach().cpu().numpy())
    cv2.imwrite(output_path + flag + "_clip2d_partial.png", (clip2d[:,:,0:3] * 255).type(dtype=torch.uint8).detach().cpu().numpy())

def generate_clip2d_SAM_SortBboxBackground_Interpolation(flag, image, output_path):
    w1 = 0.5
    sort = True
    extend = 8
    blur = (3,3,0)
    if not exists(output_path + flag + "_masks.pt"):
        sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)
        torch.save(masks, output_path + flag + "_masks.pt")
    masks = torch.load(output_path + flag + "_masks.pt", map_location = device)
    #Sort mask
    if sort:
        masks = sorted(masks, key = lambda x : x["area"], reverse=True)
    masks_number = len(masks)
    print(masks_number)
    image_h = image.shape[0]
    image_w = image.shape[1]
    clip2d = torch.zeros((image_h, image_w, 512)).to(device)
    for i in range(masks_number):
        print(i)
        #mask
        masked_image = image * np.expand_dims(masks[i]["segmentation"], -1)
        #(blur) background
        background_image = image * np.expand_dims(np.invert(masks[i]["segmentation"]), -1)
        background_image = gaussian_filter(background_image, sigma=blur)
        maskedBackground_image = masked_image + background_image
        #bbox
        bbox = masks[i]["bbox"]
        if bbox[3] + 1 == image_h and bbox[2] + 1 == image_w:
            continue
        if bbox[1] - extend >= 0 and bbox[1] + bbox[3] + extend < image_h and bbox[0] - extend >= 0 and bbox[0] + bbox[2] < image_w:
            bboxBackground_image = maskedBackground_image[bbox[1]-extend:bbox[1]+bbox[3]+extend, bbox[0]-extend:bbox[0]+bbox[2]+extend]
        else:
            bboxBackground_image = maskedBackground_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        if not exists(output_path + flag + "_masked_images"):
            os.mkdir(output_path + flag + "_masked_images")
        if not exists(output_path + flag + "_masked_images/" + flag + "_masked_image" + "_" + str(i) + ".png"):
            masked_image = cv2.cvtColor(np.array(masked_image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path + flag + "_masked_images/" + flag + "_masked_image" + "_" + str(i) + ".png", np.array(masked_image))
        if not exists(output_path + flag + "_bboxBackground_images"):
            os.mkdir(output_path + flag + "_bboxBackground_images")
        if not exists(output_path + flag + "_bboxBackground_images/" + flag + "_bboxBackground_image" + "_" + str(i) + ".png"):
            bboxBackground_image = cv2.cvtColor(np.array(bboxBackground_image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path + flag + "_bboxBackground_images/" + flag + "_bboxBackground_image" + "_" + str(i) + ".png", np.array(bboxBackground_image))
        bboxBackground_image = Image.fromarray(bboxBackground_image)
        preprocessed_bboxBackground_image = preprocess_img(bboxBackground_image).unsqueeze(0).to(device)
        with torch.no_grad():
            bboxBackground_image_clip1d = torch.squeeze(pretrained_clip_model.encode_image(preprocessed_bboxBackground_image))
        bboxBackground_image_clip1d_normalized = bboxBackground_image_clip1d.div(bboxBackground_image_clip1d.norm(p = 2).expand_as(bboxBackground_image_clip1d))
        #Interpolation
        if torch.count_nonzero(clip2d[torch.tensor(masks[i]["segmentation"])]) == 0:
            clip2d[torch.tensor(masks[i]["segmentation"])] = bboxBackground_image_clip1d_normalized
        else: #同一个mask下，可以有一半0，一半非0
            clip2d[torch.tensor(masks[i]["segmentation"])] = normalize((w1 * clip2d[torch.tensor(masks[i]["segmentation"])] + (1-w1) * bboxBackground_image_clip1d_normalized), p = 2.0, dim = -1)
    np.save(output_path + flag + "_clip2d", clip2d.detach().cpu().numpy())
    cv2.imwrite(output_path + flag + "_clip2d_partial.png", (clip2d[:,:,0:3] * 255).type(dtype=torch.uint8).detach().cpu().numpy())

def generate_clip2d_Sliding_window(flag, image, output_path):
    #get_clipmap()
    window_size = 3
    n_segments = [5]
    patch_size = 32
    target_size = 7
    upsample = 2
    target_size = target_size * upsample
    align_corners = None
    start_block = 0
    pretrained_clip_model, preprocess_img = clip.load("ViT-B/32", device=device)
    pretrained_clip_model = pretrained_clip_model.to(device)
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

if __name__ == "__main__":
    input_path = "data/"
    flag = "mic"
    single_image_num = 1
    method = "SAM_SortBboxBackground_Interpolation"
    output_path = "output_data/" + flag + "/" + method + "/"
    text = "stand"
    thr = 0.8
    image_dic = os.listdir(input_path + flag)
    for image_path in image_dic:
        image_num = image_path[:-4]
        if single_image_num != 0:
            image_num = str(single_image_num)
        image = cv2.cvtColor(cv2.imread(input_path + flag + "/" + image_num + ".png"), cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_path + flag + image_num + ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not exists(output_path + flag + image_num + "_clip2d.npy"):
            if method == "SAM_Replacement":
                generate_clip2d_SAM_Replacement(flag + image_num, image, output_path)
            elif method == "Sliding_window":
                generate_clip2d_Sliding_window(flag + image_num, image, output_path)
            elif method == "SAM_Interpolation":
                generate_clip2d_SAM_Interpolation(flag + image_num, image, output_path)
            elif method == "SAM_SortMask_Interpolation":
                generate_clip2d_SAM_SortMask_Interpolation(flag + image_num, image, output_path)
            elif method == "SAM_SortBboxBackground_Interpolation":
                generate_clip2d_SAM_SortBboxBackground_Interpolation(flag + image_num, image, output_path)
        clip2d = torch.from_numpy(np.load(output_path + flag + image_num + "_clip2d.npy"))
        clip2d_normalized = normalize(clip2d, p=2.0, dim = -1)
        with torch.no_grad():
            text_clip1d = torch.squeeze(pretrained_clip_model.encode_text(clip.tokenize([text]).to(device)))
        text_clip1d_normalized = normalize(text_clip1d, p=2.0, dim = -1)
        np.save(output_path + flag + "feature_" + text, text_clip1d_normalized.detach().cpu().numpy())
        saliency2d = torch.tensordot(clip2d_normalized.float(), text_clip1d_normalized.cpu(), dims=([2],[0])).detach().numpy()
        saliency2d = np.squeeze(saliency2d)
        saliency2d_zero_index = np.where(saliency2d == 0)
        saliency2d[saliency2d_zero_index] = 10
        saliency2d[saliency2d_zero_index] = np.min(saliency2d)
        saliency2d_remapped = (saliency2d - np.min(saliency2d)) / (np.max(saliency2d) - np.min(saliency2d))
        plt.imsave(output_path + flag + image_num + "_" + text + "_" + "saliency2d.png", saliency2d_remapped)
        indices = np.where(saliency2d_remapped >= thr)
        x_y_coords =list(zip(indices[0], indices[1]))
        saliency2d_remapped_thr = np.zeros(saliency2d_remapped.shape)
        for index in x_y_coords:
            saliency2d_remapped_thr[index] = saliency2d[index]
        plt.imsave(output_path + flag + image_num + "_" + text + "_" + "saliency2d" + "_" + str(thr) + ".png", saliency2d_remapped_thr)
        if single_image_num != 0:
            break