import network
import utils
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from datasets import Cityscapes
import numpy as np
import json
 
def labelgenerator(imagefilepaths, model, ckpt, bucket_idx = 0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    dir_name = "output/weaklabels/KITTI-360/bucket_" + str(bucket_idx) + "/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    results = []  # To store paths for the JSON file
    
    model = network.modeling.__dict__[model](num_classes=19, output_stride=16)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    if ckpt is not None and os.path.isfile(ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

        #denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    transform = T.Compose([T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                            ])
    # print(f"opts.crop_val: {opts.crop_val}, opts.crop_size: {opts.crop_size}")

    with torch.no_grad():
        model = model.eval()
        entropy_values = []
        confidence_values = []
        
        for img_path in tqdm(imagefilepaths):
            print(f"Processing image: {img_path}")
            ext = os.path.basename(img_path).split('.')[-1]
            print(f"ext: {ext}")
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0) # To tensor of NCHW
            img = img.to(device)
            
            # print(f"img.shape: {img.shape}")
            output = model(img)
            # print(f"output.shape: {output.shape}")

            # for i in range(output.shape[1]):
            #     print(f"output[{i}].shape: {output[0, i]}")

            pred = output.max(1)[1].cpu().numpy()[0] # HW
            # print(f"pred.shape: {pred.shape}")
            # print(pred)
            
            # Calculate confidence and average entropy for this image
            prob = torch.softmax(output, dim=1)
            confidence = prob.max(1)[0].mean().item()
            entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)  # HW
            avg_entropy = entropy.mean().item()
            entropy_values.append(avg_entropy)      
            confidence_values.append(confidence)     

            colorized_preds = Cityscapes.decode_target(pred).astype('uint8')
            colorized_preds = Image.fromarray(colorized_preds)
            # colorized_preds = Image.fromarray(pred.astype('uint8'))

            label_path = os.path.join(dir_name, img_name + '.png')
            colorized_preds.save(label_path)
            
            # Add the image path and label path to the results list
            results.append({
                'image_path': img_path,
                'label_path': label_path
            })

            
        overall_avg_entropy = np.mean(entropy_values)
        overall_avg_confidence = np.mean(confidence_values)
        print(f"Overall average entropy for all images: {overall_avg_entropy:.4f}")
        print(f"Overall average confidence for all images: {overall_avg_confidence:.4f}")

        # Save results to a JSON file
        filename = "image_label_bucket_" + str(bucket_idx) + ".json"
        json_path = os.path.join(dir_name, filename)
        with open(json_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        print(f"Saved image and label paths to {json_path}")