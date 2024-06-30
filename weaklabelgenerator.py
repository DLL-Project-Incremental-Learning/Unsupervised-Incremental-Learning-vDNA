import os
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as T
from datasets import Cityscapes
import numpy as np
import json

def load_model(model, ckpt, device):
    if ckpt is not None and os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print(f"Resume model from {ckpt}")
    else:
        raise FileNotFoundError(f"Checkpoint file {ckpt} not found.")
    return model

def process_image(img_path, model, transform, device, dir_name, results):
    ext = os.path.basename(img_path).split('.')[-1]
    img_name = os.path.basename(img_path)[:-len(ext)-1]
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return

    img = transform(img).unsqueeze(0).to(device)  # To tensor of NCHW
    output = model(img)  # Forward pass
    pred = output.max(1)[1].cpu().numpy()[0]  # HW

    # Calculate confidence and average entropy for this image
    prob = torch.softmax(output, dim=1)
    confidence = prob.max(1)[0].mean().item()
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1).mean().item()

    # colorized_preds = Cityscapes.decode_target(pred).astype('uint8')
    # colorized_preds = Image.fromarray(colorized_preds)
    colorized_preds = Image.fromarray(pred.astype('uint8'))

    label_path = os.path.join(dir_name, img_name + '.png')
    colorized_preds.save(label_path)

    # Add the image path and label path to the results list
    results.append({
        'image_path': img_path,
        'label_path': label_path,
        'entropy': entropy,
        'confidence': confidence
    })

def labelgenerator(imagefilepaths, model, ckpt, bucket_idx=0, val=True, order="asc"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    if not val:
        dir_name = f"outputs/weaklabels/KITTI-360/{order}/bucket_{bucket_idx}/"
    else:
        dir_name = f"outputs/weaklabels/KITTI-360/{order}/val_bucket_{bucket_idx}/"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    results = []  # To store paths for the JSON file
    
    # Load model
    model = load_model(model, ckpt, device)
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        model.eval()
        
        for img_path in tqdm(imagefilepaths):
            process_image(img_path, model, transform, device, dir_name, results)

        # Compute overall averages
        entropy_values = [item['entropy'] for item in results]
        confidence_values = [item['confidence'] for item in results]
        overall_avg_entropy = np.mean(entropy_values)
        overall_avg_confidence = np.mean(confidence_values)
        
        print(f"Overall average entropy for all images: {overall_avg_entropy:.4f}")
        print(f"Overall average confidence for all images: {overall_avg_confidence:.4f}")
        
        # Save results to a JSON file
        if not val:
            filename = f"image_label_bucket_{bucket_idx}.json"
        else:
            filename = f"image_label_val_bucket_{bucket_idx}.json"
        json_path = os.path.join(dir_name, filename)
        with open(json_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        print(f"Saved image and label paths to {json_path}")
    
    return dir_name


# def labelgenerator(imagefilepaths, model, ckpt, bucket_idx=0):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Device: %s" % device)
    # dir_name = f"output/weaklabels/KITTI-360/bucket_{bucket_idx}/"
    # if not os.path.exists(dir_name):
    #    os.makedirs(dir_name)

    # results = []  # To store paths for the JSON file
    
    #if ckpt is not None and os.path.isfile(ckpt):
    #    checkpoint = torch.load(ckpt, map_location=device)
    #    model.load_state_dict(checkpoint["model_state"])
    #    model = nn.DataParallel(model)
    #    model.to(device)
    #    print(f"Resume model from {ckpt}")
    #else:
    #    raise FileNotFoundError(f"Checkpoint file {ckpt} not found.")
    
    #transform = T.Compose([
    #    T.ToTensor(),
    #    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #])

    #with torch.no_grad():
        #model.eval()
        #entropy_values = []
        #confidence_values = []
        
        #for img_path in tqdm(imagefilepaths):
            #print(f"Processing image: {img_path}")
            # ext = os.path.basename(img_path).split('.')[-1]
            # img_name = os.path.basename(img_path)[:-len(ext)-1]
            # img = Image.open(img_path).convert('RGB')
            # img = transform(img).unsqueeze(0)  # To tensor of NCHW
            # img = img.to(device)
            
            # output = model(img)
            # pred = output.max(1)[1].cpu().numpy()[0]  # HW
            
            # Calculate confidence and average entropy for this image
            # prob = torch.softmax(output, dim=1)
            # confidence = prob.max(1)[0].mean().item()
            # entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)  # HW
            # avg_entropy = entropy.mean().item()
            # entropy_values.append(avg_entropy)
            # confidence_values.append(confidence)
            
            # colorized_preds = Cityscapes.decode_target(pred).astype('uint8')
            # colorized_preds = Image.fromarray(colorized_preds)
            
            # label_path = os.path.join(dir_name, img_name + '.png')
            # colorized_preds.save(label_path)
            
            # Add the image path and label path to the results list
            # results.append({
            #   'image_path': img_path,
            #   'label_path': label_path
            #})
        
        # overall_avg_entropy = np.mean(entropy_values)
        # overall_avg_confidence = np.mean(confidence_values)
        # print(f"Overall average entropy for all images: {overall_avg_entropy:.4f}")
        # print(f"Overall average confidence for all images: {overall_avg_confidence:.4f}")
        
        # Save results to a JSON file
        #filename = f"image_label_bucket_{bucket_idx}.json"
        #json_path = os.path.join(dir_name, filename)
        #with open(json_path, 'w') as json_file:
        #    json.dump(results, json_file, indent=4)
        #print(f"Saved image and label paths to {json_path}")
