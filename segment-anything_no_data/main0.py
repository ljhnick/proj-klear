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

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_clip_model, preprocess_img = clip.load("ViT-B/32", device=device)
pretrained_clip_model = pretrained_clip_model.to(device)
def show_mask(i, mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    plt.savefig("mask_" + str(i))
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
def compute_logits(image_features, text_features):
    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    # cosine similarity as logits
    logit_scale = 100
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    return logits_per_image, logits_per_text


token = "all"
if token != "all":
    predictor = SamPredictor(build_sam(checkpoint="sam_vit_h_4b8939.pth"))
    image = cv2.imread("notebooks/images/car2.jpg") 
    #image = cv2.imread("segment_anything/Reproject_CLIP/flickr8k/Images/667626_18933d713e.jpg") 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    if token == "point":
        input_point = np.array([[500, 375], [1125, 625]])
        input_label = np.array([1, 1])
        masks, scores, logits = predictor.predict_text(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(i, mask, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
    elif token == "text":
        input_text = np.array(["car"])
        masks, scores, logits = predictor.predict_text(
            texts = input_text,
            multimask_output=True,
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(i, mask, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()
    

def generate_trainSet_clip_imgFeatures(image_dic_path, output, mask_number_flag, scale_percent, gen_type, scale_percent2, combine_flag, w1, w2):
    if output == "mac":
        output_path = "output_data/" + gen_type + "/"
    elif output == "linux":
        output_path = "/users/aren10/data/datasets/seg_all_2DCLIP_gt/mic/train/"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    if gen_type == "multi_layer":
        mask_generator_new = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.5,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )
    image_dic = os.listdir(image_dic_path)
    for filename in image_dic:
        print(filename)
        #image = cv2.imread("notebooks/images/dog.jpg")  #image = cv2.imread("segment_anything/Reproject_CLIP/flickr8k/Images/667626_18933d713e.jpg") 
        image = cv2.imread(image_dic_path + filename)
        cv2.imwrite(output_path + filename[:-4] + ".png", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if gen_type == "multi_res":
            width2 = int(image.shape[1] * scale_percent2)
            height2 = int(image.shape[0] * scale_percent2)
            dim2 = (width2, height2)
            image2 = cv2.resize(image, dim2) #80 * 80
        width = int(image.shape[1] * scale_percent)
        height = int(image.shape[0] * scale_percent)
        dim = (width, height)
        image = cv2.resize(image, dim) #320 * 320
        masks = mask_generator.generate(image) #can sort the masks by area, and disregard any mask smaller than certain area
        print(len(masks))
        #print(masks)
        if gen_type == "multi_layer":
            masks_new = mask_generator_new.generate(image)
            print(len(masks_new))
        if gen_type == "multi_res":
            masks2 = mask_generator.generate(image2)
            print(len(masks2))
        clip2d_gt = torch.zeros((image.shape[0], image.shape[1], 512))
        clip2d_gt = clip2d_gt.to(device)
        if gen_type == "multi_layer":
            clip2d_gt_new = torch.zeros((image.shape[0], image.shape[1], 512))
            clip2d_gt_new = clip2d_gt_new.to(device)
        if gen_type == "multi_res":
            clip2d_gt2 = torch.zeros((image2.shape[0], image2.shape[1], 512))
            clip2d_gt2 = clip2d_gt2.to(device)
        if mask_number_flag == 0:
            mask_number = len(masks)
            if gen_type == "multi_layer":
                mask_number_new = len(masks_new)
            if gen_type == "multi_res":
                mask_number2 = len(masks2)
        else:
            mask_number = mask_number_flag
            if gen_type == "multi_layer":
                mask_number_new = mask_number_flag
            if gen_type == "multi_res":
                mask_number2 = mask_number_flag
        if gen_type == "masked":
            for i in range(mask_number):
                print(i)
                masked_image = torch.zeros(image.shape)
                masked_image = torch.from_numpy(image) * torch.unsqueeze(torch.from_numpy(masks[i]["segmentation"]), -1) # multiply both by bbx XYWH
                masked_image = masked_image.type(dtype=torch.uint8)
                if i == 2:
                    cv2.imwrite("masked_image.png", np.array(masked_image))
                #clip image 1D vector
                masked_image = cv2.cvtColor(masked_image.detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                masked_image = Image.fromarray(masked_image)
                masked_image = preprocess_img(masked_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    masked_image_features = pretrained_clip_model.encode_image(masked_image)
                #clip image 2D map
                masked_image_features = torch.unsqueeze(masked_image_features, 0)
                matrix = torch.zeros(clip2d_gt.shape)
                segmentation = torch.from_numpy(masks[i]["segmentation"]).to(device).type(torch.uint8)
                for j in range(matrix.shape[2]):
                    matrix[:,:,j] = segmentation
                if i == 2:
                    #print(matrix[89,169,0])
                    #print(matrix[89,169,1])
                    print(matrix[0,0,0])
                    print(matrix[0,0,1])
                    print(masked_image_features[0][0][0:3])
                    print((matrix * masked_image_features)[0, 0, 0:3])
                    #print((matrix * masked_image_features)[89, 169, 0:3])
                #ones = torch.ones()
                clip2d_gt[masks[i]["segmentation"]] = masked_image_features
                if i == 2:
                    #print(clip2d_gt[89, 169, 0:3])
                    print("~")
        elif gen_type == "bbox":
            for i in range(mask_number):
                print(i)
                masked_image = torch.zeros(image.shape)
                masked_image = torch.from_numpy(image) * torch.unsqueeze(torch.from_numpy(masks[i]["segmentation"]), -1) # multiply both by bbx XYWH
                masked_image = masked_image.type(dtype=torch.uint8)
                bbox = masks[i]["bbox"]
                bbox_image = masked_image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                #clip image 1D vector
                bbox_image = cv2.cvtColor(bbox_image.detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                bbox_image = Image.fromarray(bbox_image)
                bbox_image = preprocess_img(bbox_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    bbox_image_features = pretrained_clip_model.encode_image(bbox_image)
                #clip image 2D map
                bbox_image_features = torch.unsqueeze(bbox_image_features, 0)
                clip2d_gt[masks[i]["segmentation"]] = bbox_image_features
                if i == 2:
                    #print(clip2d_gt[89, 169, 0:3])
                    print("~")
        elif gen_type == "multi_layer":
            for i in range(mask_number):
                print(i)
                masked_image = torch.zeros(image.shape)
                masked_image = torch.from_numpy(image) * torch.unsqueeze(torch.from_numpy(masks[i]["segmentation"]), -1) # multiply both by bbx XYWH
                masked_image = masked_image.type(dtype=torch.uint8)
                #clip image 1D vector
                masked_image = cv2.cvtColor(masked_image.detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                masked_image = Image.fromarray(masked_image)
                masked_image = preprocess_img(masked_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    masked_image_features = pretrained_clip_model.encode_image(masked_image)
                #clip image 2D map
                masked_image_features = torch.unsqueeze(masked_image_features, 0)
                matrix = torch.zeros(clip2d_gt.shape)
                segmentation = torch.from_numpy(masks[i]["segmentation"]).to(device).type(torch.uint8)
                for j in range(matrix.shape[2]):
                    matrix[:,:,j] = segmentation
                #ones = torch.ones()
                clip2d_gt[masks[i]["segmentation"]] = masked_image_features
            for j in range(mask_number_new):
                print(j)
                masked_image = torch.zeros(image.shape)
                masked_image = torch.from_numpy(image) * torch.unsqueeze(torch.from_numpy(masks_new[j]["segmentation"]), -1) # multiply both by bbx XYWH
                masked_image = masked_image.type(dtype=torch.uint8)
                #clip image 1D vector
                masked_image = cv2.cvtColor(masked_image.detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                masked_image = Image.fromarray(masked_image)
                masked_image = preprocess_img(masked_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    masked_image_features = pretrained_clip_model.encode_image(masked_image)
                #clip image 2D map
                masked_image_features = torch.unsqueeze(masked_image_features, 0)
                matrix = torch.zeros(clip2d_gt_new.shape)
                segmentation = torch.from_numpy(masks_new[j]["segmentation"]).to(device).type(torch.uint8)
                for k in range(matrix.shape[2]):
                    matrix[:,:,k] = segmentation
                #ones = torch.ones()
                clip2d_gt_new[masks_new[j]["segmentation"]] = masked_image_features
            if combine_flag == "avg":
                clip2d_gt = (w1 * clip2d_gt + w2 * clip2d_gt_new)
            elif combine_flag == "concat":
                clip2d_gt = torch.cat((clip2d_gt, clip2d_gt_new), 2)
            print(clip2d_gt.shape)

        elif gen_type == "multi_res":
            for i in range(mask_number):
                print(i)
                masked_image = torch.zeros(image.shape)
                masked_image = torch.from_numpy(image) * torch.unsqueeze(torch.from_numpy(masks[i]["segmentation"]), -1) # multiply both by bbx XYWH
                masked_image = masked_image.type(dtype=torch.uint8)
                #clip image 1D vector
                masked_image = cv2.cvtColor(masked_image.detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                masked_image = Image.fromarray(masked_image)
                masked_image = preprocess_img(masked_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    masked_image_features = pretrained_clip_model.encode_image(masked_image)
                #clip image 2D map
                masked_image_features = torch.unsqueeze(masked_image_features, 0)
                matrix = torch.zeros(clip2d_gt.shape)
                segmentation = torch.from_numpy(masks[i]["segmentation"]).to(device).type(torch.uint8)
                for j in range(matrix.shape[2]):
                    matrix[:,:,j] = segmentation
                #ones = torch.ones()
                clip2d_gt[masks[i]["segmentation"]] = masked_image_features
            for j in range(mask_number2):
                print(j)
                masked_image = torch.zeros(image2.shape)
                masked_image = torch.from_numpy(image2) * torch.unsqueeze(torch.from_numpy(masks2[j]["segmentation"]), -1) # multiply both by bbx XYWH
                masked_image = masked_image.type(dtype=torch.uint8)
                #clip image 1D vector
                masked_image = cv2.cvtColor(masked_image.detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                masked_image = Image.fromarray(masked_image)
                masked_image = preprocess_img(masked_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    masked_image_features = pretrained_clip_model.encode_image(masked_image)
                #clip image 2D map
                masked_image_features = torch.unsqueeze(masked_image_features, 0)
                matrix = torch.zeros(clip2d_gt2.shape)
                segmentation = torch.from_numpy(masks2[j]["segmentation"]).to(device).type(torch.uint8)
                for k in range(matrix.shape[2]):
                    matrix[:,:,k] = segmentation
                #ones = torch.ones()
                clip2d_gt2[masks2[j]["segmentation"]] = masked_image_features
            r2 = int(clip2d_gt.shape[0]) 
            c2 = int(clip2d_gt.shape[1])
            print(clip2d_gt.shape)
            print(clip2d_gt2.shape)
            clip2d_gt2 = cv2.resize(np.array(clip2d_gt2), (c2,r2))
            print(clip2d_gt2.shape)
            if combine_flag == "avg":
                clip2d_gt2 = torch.from_numpy(clip2d_gt2)
                clip2d_gt = (w1 * clip2d_gt + w2 * clip2d_gt2)
            elif combine_flag == "concat":
                clip2d_gt = torch.cat((clip2d_gt, torch.from_numpy(clip2d_gt2)), 2)
            print(clip2d_gt.shape)
        
        elif gen_type == "lerf":
            return

        np.save(output_path + filename[:-4] + "_clip2d_gt", clip2d_gt.detach().cpu().numpy())
        clip2d_gt_partial = (clip2d_gt[:,:,0:3] * 255).type(dtype=torch.uint8)
        cv2.imwrite(output_path + filename[:-4] + "_clip2d_gt_partial.png", cv2.cvtColor(clip2d_gt_partial.detach().cpu().numpy(), cv2.COLOR_RGB2BGR))
        exit(0)


#test: Top Left Corner is (0,0), xnview(y,x)
def test(load, file, image_features1_x, image_features1_y, image_features2_x, image_features2_y, text, scale_percent):
    if load == "mac":
        load_path = "output_data/" + file
    elif load == "linux":
        load_path = "/users/aren10/data/datasets/seg_all_2DCLIP_gt/" + file
    clip2d_gt = torch.from_numpy(np.load(load_path))
    print(clip2d_gt.shape)
    image_features1 = torch.unsqueeze(clip2d_gt[(int)(image_features1_x * scale_percent), (int)(image_features1_y * scale_percent)], 0).to(torch.float32).to(device)
    image_features2 = torch.unsqueeze(clip2d_gt[(int)(image_features2_x * scale_percent), (int)(image_features2_y * scale_percent)], 0).to(torch.float32).to(device)
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = pretrained_clip_model.encode_text(text)
        logits_per_image, logits_per_text = compute_logits(image_features1.float(), text_features.float())
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("Label probs:", probs)
        logits_per_image, logits_per_text = compute_logits(image_features2.float(), text_features.float())
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        print("Label probs:", probs)



def saliency_map(image_path = None, texts = None, flag = "image", image_feature_path = None, img_x = None, img_y = None):
    if flag == "image":
        #Encode image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(image[89,168,0])
        image = Image.fromarray(image)
        image = preprocess_img(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = pretrained_clip_model.encode_image(image)
        image_feature = torch.squeeze(image_feature)
        print(image_feature[0:3])
    elif flag == "npy":
        image_feature = torch.from_numpy(np.load(image_feature_path)[img_x, img_y])
        print(image_feature[0:3])
    #Encode text
    texts = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_features = pretrained_clip_model.encode_text(texts)
    #Saliency
    saliency_map = []
    for i in range(len(text_features)):
        text_feature = text_features[i]
        text_feature = torch.squeeze(text_feature)
        saliency_map.append(torch.dot(text_feature, image_feature))
    print(saliency_map)






if __name__ == "__main__":
    flag = "lerf"
    combine_flag = "concat"
    generate_trainSet_clip_imgFeatures("notebooks/images/dog/", "mac", 0, 1.0, flag, 0.5, combine_flag, 0.5, 0.5)
    #test("mac", "r_80_clip2d_gt.npy", 159, 223, 569 ,383, ["mic", "line", "iron stand"], 0.4) # Size = 4 * 800 * 800 * 512 = 1.31 Gb vs Size = 4 * 320 * 320 * 768 = 0.20 Gb
    #saliency_map("1masked_image.png", ["mic","wire"])
    #saliency_map("2masked_image.png", ["mic","wire"])
    #saliency_map("1bbox_image.png", ["mic","wire"])
    #saliency_map("2bbox_image.png", ["mic","wire"])
    #saliency_map(None, ["mic","wire"], "npy", "output_data/masked/r_80_clip2d_gt.npy", 85, 156)
    #saliency_map(None, ["mic","wire"], "npy", "output_data/masked/r_80_clip2d_gt.npy", 151, 265)
    #saliency_map(None, ["mic","wire"], "npy", "output_data/bbox/r_80_clip2d_gt.npy", 85, 156)
    #saliency_map(None, ["mic","wire"], "npy", "output_data/bbox/r_80_clip2d_gt.npy", 151, 265)
    #exit(0)
    text1 = "legs with jeans"
    text2 = "massage machine"
    thr = 0.95
    root_path = "output_data/" + flag + "/"
    image_features_all = np.load(root_path + "dog_clip2d_gt.npy")
    image_features_all = torch.from_numpy(image_features_all)
    with torch.no_grad():
        r = image_features_all.shape[0]
        c = image_features_all.shape[1]
        f = image_features_all.shape[2]
        input = torch.empty(r, c, 1)
        query_map1 = torch.zeros_like(input)
        query_map2 = torch.zeros_like(input)
        text_tokenized1 = clip.tokenize([text1])#.cuda()
        text_features1 = torch.squeeze(pretrained_clip_model.encode_text(text_tokenized1)) #torch.Size([512])
        if (flag == "multi_layer" or flag == "multi_res") and (flag == "concat"):
            text_features1 = torch.cat((text_features1, text_features1), 0)
        text_tokenized2 = clip.tokenize([text2])#.cuda()
        text_features2 = torch.squeeze(pretrained_clip_model.encode_text(text_tokenized2)) #torch.Size([512])
        if (flag == "multi_layer" or flag == "multi_res") and (flag == "concat"):
            text_features2 = torch.cat((text_features2, text_features2), 0)
        for i in range(r):
            for j in range(c):
                image_features = image_features_all[i,j,:]
                #image_features_normalized = image_features / image_features.norm(keepdim=True)
                #text_features_normalized1 = text_features1 / text_features1.norm(keepdim=True)
                #text_features_normalized2 = text_features2 / text_features1.norm(keepdim=True)
                val1 = torch.dot(text_features1, image_features)
                val2 = torch.dot(text_features2, image_features)
                query_map1[i][j] = val1
                query_map2[i][j] = val2
    query_map1 = ((query_map1 - torch.min(query_map1)) / (torch.max(query_map1) - torch.min(query_map1)))
    for i in range(r):
        for j in range(c):
            if query_map1[i][j] < thr:
                query_map1[i][j] = 0
    cv2.imwrite(root_path + text1 + "_heat.png", np.array(query_map1 * 255))
    #exit(0)
    query_map2 = ((query_map2 - torch.min(query_map2)) / (torch.max(query_map2) - torch.min(query_map2)))
    for i in range(r):
        for j in range(c):
            if query_map2[i][j] < thr:
                query_map2[i][j] = 0
    cv2.imwrite(root_path + text2 + "_heat.png", np.array(query_map2 * 255))
    











