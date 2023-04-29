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
device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_clip_model, preprocess_img = clip.load("ViT-B/32", device=device)
pretrained_clip_model = pretrained_clip_model.to(device)
def generate_clip2d(input_path, output_path, flag):
    #Read image
    image = cv2.imread(input_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    """
    #Generate masks of the image
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    torch.save(masks, output_path + flag + "_masks.pt")
    """
    #Load masks of the image
    masks = torch.load(output_path + flag + "_masks.pt", map_location = device)
    masks_number = len(masks)
    print(masks_number)
    #Generate clip2d
    image_h = image.shape[0]
    image_w = image.shape[1]
    clip2d = torch.zeros((image_h, image_w, 512)).to(device)
    for i in range(masks_number):
        print(i)
        """
        #Test
        i = 2
        """
        masked_image = image * np.expand_dims(masks[i]["segmentation"], -1)
        
        """
        #Test
        masked_image = cv2.cvtColor(np.array(masked_image), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path + flag + str(i) + "_masked_image.png", np.array(masked_image))
        exit(0)
        """
        
        masked_image = Image.fromarray(masked_image)
        preprocessed_masked_image = preprocess_img(masked_image).unsqueeze(0).to(device)
        with torch.no_grad():
            masked_image_clip1d = torch.squeeze(pretrained_clip_model.encode_image(preprocessed_masked_image))
        masked_image_clip1d_normalized = masked_image_clip1d.div(masked_image_clip1d.norm(p = 2).expand_as(masked_image_clip1d))
        """
        #Test
        x = torch.tensor([0.1,0])
        x_normalized = x.div(x.norm(p=2).expand_as(x))
        print(x_normalized)
        y = torch.tensor([0.1,0])
        y_normalized = y.div(y.norm(p=2).expand_as(y))
        print(y_normalized)
        score = torch.dot(x, y)
        print(score)
        score = torch.dot(x_normalized, y_normalized)
        print(score)
        exit(0)
        """
        """
        #Test
        text_clip1d = torch.squeeze(pretrained_clip_model.encode_text(clip.tokenize(["human legs with blue pants"])))
        text_clip1d_normalized = text_clip1d.div(text_clip1d.norm(p=2).expand_as(text_clip1d))
        score = torch.dot(masked_image_clip1d_normalized, text_clip1d_normalized)
        print(score)
        exit(0)
        """
        """
        #Test
        m = torch.zeros((3,3,2))
        print(m)
        print(m.shape)
        flag = torch.tensor([[True, False, True], [True, False, True], [True, False, True]])
        print(flag)
        print(flag.shape)
        m[flag] = torch.tensor([1.,2.])
        print(m)
        exit(0)
        """
        masked_image_clip1d_final = (clip2d[torch.tensor(masks[i]["segmentation"])] + masked_image_clip1d_normalized)[0]
        masked_image_clip1d_final_normalized = masked_image_clip1d_final.div(masked_image_clip1d_final.norm(p = 2).expand_as(masked_image_clip1d_final))
        clip2d[torch.tensor(masks[i]["segmentation"])] = masked_image_clip1d_final_normalized
        """"
        #Test
        x_normalized = clip2d[52,753]
        text_clip1d = torch.squeeze(pretrained_clip_model.encode_text(clip.tokenize(["human legs with blue pants"])))
        text_clip1d_normalized = text_clip1d.div(text_clip1d.norm(p=2).expand_as(text_clip1d))
        score = torch.dot(x_normalized, text_clip1d_normalized)
        print(score)
        print(x_normalized.dtype)
        print(text_clip1d_normalized.dtype)
        print(x_normalized[0:3])
        print(text_clip1d_normalized[0:3])
        exit(0)
        """
    #save clip2d
    np.save(output_path + flag + "_clip2d", clip2d.detach().cpu().numpy())
    cv2.imwrite(output_path + flag + "_clip2d_partial.png", (clip2d[:,:,0:3] * 255).type(dtype=torch.uint8).detach().cpu().numpy())




if __name__ == "__main__":
    flag = "dog"
    input_path = "data/" + flag + ".png"
    output_path = "output_data/"
    #generate_clip2d(input_path, output_path, flag) #SAM is multi-level by nature (Future work: You can first sort mask by its area, then add overlapped clip with smaller mask area by small factor eg 0.2)
    clip2d = torch.from_numpy(np.load(output_path + flag + "_clip2d.npy"))

    #Test (thr > 0.25)
    #1. (435, 180, white dog 0.2645 > dog 0.2636 > white pug 0.2593 > white Shiba Inu 0.2290 > cat 0.2134)
    #2. (211, 713, human legs with blue pants 0.3019 > human legs without blue pants 0.2831 > pants 0.2466 > dog 0.2168)
    #3. (229, 600, silver round bowl 0.2739 > bowl 0.2683)
    text = "human feet"
    with torch.no_grad():
        text_clip1d = torch.squeeze(pretrained_clip_model.encode_text(clip.tokenize([text])))
    text_clip1d_normalized = text_clip1d.div(text_clip1d.norm(p=2).expand_as(text_clip1d))
    thr = 0.26
    saliency2d = torch.zeros((clip2d.shape[0], clip2d.shape[1], 1))
    for i in range(clip2d.shape[0]):
        for j in range(clip2d.shape[1]):
            """
            #Test
            i = 435
            j = 180
            """
            x_normalized = clip2d[i,j]
            score = torch.dot(x_normalized, text_clip1d_normalized)
            if score > thr:
                saliency2d[i,j] = score
            """
            #Test
            print(score)
            exit(0)
            """
    test_point = [395, 400, 111, 115]
    """
    #Way1: Non-normalized
    saliency2d = torch.squeeze(saliency2d)
    saliency2d_255 = (saliency2d * 255).type(torch.uint8)
    #Test
    print(saliency2d[test_point[0]:test_point[1], test_point[2]:test_point[3]])
    print(saliency2d_255[test_point[0]:test_point[1], test_point[2]:test_point[3]])
    #Save saliency2d
    cv2.imwrite(output_path + flag + "_" + text + "_" + "saliency2d.png", saliency2d_255.detach().numpy())
    """
    """
    #Test
    m = torch.tensor([[0.2,0],[0,0.3]])
    m_h = m.shape[0]
    m_w = m.shape[1]
    m_reshaped = torch.reshape(m, (m_h * m_w, 1))
    print(m_reshaped)
    print(m_reshaped.shape)
    m_reshaped_normalized = normalize(m_reshaped, axis = 0, norm = 'l2')
    print(m_reshaped_normalized)
    m_normalized = torch.reshape(torch.from_numpy(m_reshaped_normalized), (m_h, m_w))
    print(m_normalized)
    exit(0)
    """
    """
    #Way2: Normalized
    saliency2d = torch.squeeze(saliency2d)
    saliency2d_h = saliency2d.shape[0]
    saliency2d_w = saliency2d.shape[1]
    saliency2d_reshaped = torch.reshape(saliency2d, (saliency2d_h * saliency2d_w, 1))
    saliency2d_reshaped_normalized = normalize(saliency2d_reshaped, axis = 0, norm = "l2")
    saliency2d_normalized = torch.reshape(torch.from_numpy(saliency2d_reshaped_normalized), (saliency2d_h, saliency2d_w))
    saliency2d_normalized_255 = (saliency2d_normalized * 255).type(torch.uint8)
    #Test
    print(saliency2d[test_point[0]:test_point[1], test_point[2]:test_point[3]])
    print(saliency2d_normalized[test_point[0]:test_point[1], test_point[2]:test_point[3]])
    print(saliency2d_normalized_255[test_point[0]:test_point[1], test_point[2]:test_point[3]])
    #Save saliency2d
    cv2.imwrite(output_path + flag + "_" + text + "_" + "saliency2d.png", saliency2d_normalized_255.detach().numpy())
    """
    """
    #Test
    m = torch.tensor([[0.2,0],[0,0.3]])
    m_h = m.shape[0]
    m_w = m.shape[1]
    m_reshaped = torch.reshape(m, (m_h * m_w, 1))
    m_reshaped_normalized = (m_reshaped - torch.min(m_reshaped)) / (torch.max(m_reshaped) - torch.min(m_reshaped))
    m_normalized = torch.reshape(m_reshaped_normalized, (m_h, m_w))
    """
    #Way3: Normalized
    saliency2d = torch.squeeze(saliency2d)
    saliency2d_h = saliency2d.shape[0]
    saliency2d_w = saliency2d.shape[1]
    saliency2d_reshaped = torch.reshape(saliency2d, (saliency2d_h * saliency2d_w, 1))
    saliency2d_reshaped_normalized = (saliency2d_reshaped - torch.min(saliency2d_reshaped)) / (torch.max(saliency2d_reshaped) - torch.min(saliency2d_reshaped))
    saliency2d_normalized = torch.reshape(saliency2d_reshaped_normalized, (saliency2d_h, saliency2d_w))
    saliency2d_normalized_255 = (saliency2d_normalized * 255).type(torch.uint8)
    #Test
    print(saliency2d[test_point[0]:test_point[1], test_point[2]:test_point[3]])
    print(saliency2d_normalized[test_point[0]:test_point[1], test_point[2]:test_point[3]])
    print(saliency2d_normalized_255[test_point[0]:test_point[1], test_point[2]:test_point[3]])
    #Save saliency2d
    cv2.imwrite(output_path + flag + "_" + text + "_" + "saliency2d.png", saliency2d_normalized_255.detach().numpy())


    