import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.ops import nms
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import joblib 
from sklearn.pipeline import Pipeline
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import os
import sys

try:
    from src.feature_fusion_helper import FeatureFusionHelperTorch
except ImportError:
    from feature_fusion_helper import FeatureFusionHelperTorch

# ==========================================
# 1. CONFIGURATION
# ==========================================
LESION_INFO = {
    0: ("Actinic Keratoses", "Malignant"),
    1: ("Basal Cell Carcinoma", "Malignant"),
    2: ("Benign Keratosis", "Benign"),
    3: ("Dermatofibroma", "Benign"),
    4: ("Melanoma", "Malignant"),
    5: ("Melanocytic Nevi", "Benign"),
    6: ("Vascular Lesion", "Benign")
}
LESION_CLASSES = [v[0] for k, v in LESION_INFO.items()]

FUSION_WEIGHTS = {"ResNet50": 0.2616, "VGG19": 0.0931, "ANN": 0.1486, "SVM": 0.1846, "KNN": 0.3121}
BOX_COLORS = ["#FF3838", "#2C99A8", "#FF701F", "#64748B", "#CFD231", "#48F90A", "#92CC17"]

SEG_COLORS = {
    "SegFormer": (239, 68, 68, 140),       # Red
    "Attention U-Net": (59, 130, 246, 140), # Blue
    "LightweightUNet": (34, 197, 94, 140),    # Green
    "DeepLabV3+": (234, 179, 8, 140),   # Yellow
    "Fusion (Top 3)": (168, 85, 247, 160) # Purple
}

# ==========================================
# 2. ARCHITECTURES (Custom LightweightUNet)
# ==========================================
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = ConvBNReLU(in_ch + skip_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1); x1 = F.pad(x1, [x2.size(3)-x1.size(3), 0, x2.size(2)-x1.size(2), 0])
        return self.conv(torch.cat([x2, x1], dim=1))

class LightweightUNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder = models.resnet34(weights=None) 
        self.enc_conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu)
        self.enc_pool = self.encoder.maxpool
        self.enc_layer1 = self.encoder.layer1; self.enc_layer2 = self.encoder.layer2
        self.enc_layer3 = self.encoder.layer3; self.enc_layer4 = self.encoder.layer4
        self.aspp = ConvBNReLU(512, 256, kernel_size=1, padding=0)
        self.up4 = UpBlock(256, 256, 256); self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64); self.up1 = UpBlock(64, 64, 64)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
    def forward(self, x):
        x0 = self.enc_conv1(x); x1 = self.enc_pool(x0); x1 = self.enc_layer1(x1)
        x2 = self.enc_layer2(x1); x3 = self.enc_layer3(x2); x4 = self.enc_layer4(x3)
        return torch.sigmoid(F.interpolate(self.final_conv(self.up1(self.up2(self.up3(self.up4(self.aspp(x4), x3), x2), x1), x0)), scale_factor=2, mode='bilinear', align_corners=True))

import __main__
setattr(__main__, "LightweightUNet", LightweightUNet)
setattr(__main__, "ConvBNReLU", ConvBNReLU)
setattr(__main__, "UpBlock", UpBlock)


# ==========================================
# 3. MODEL MANAGER
# ==========================================
class ModelManager:
   class ModelManager:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loaded_models = {}
        
        # --- PATH FIX: Get absolute path to the project root ---
        # This ensures it works on Streamlit Cloud regardless of working directory
        current_file_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file_path) # .../src
        project_root = os.path.dirname(src_dir)      # .../ (root)
        
        def get_path(rel_path):
            return os.path.join(project_root, rel_path)

        # Classification Helper
        self.cnn_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.cnn_model.fc = nn.Identity(); self.cnn_model.to(self.device).eval()
        self.cnn_transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.fusion_helper = FeatureFusionHelperTorch(
            cnn_model=self.cnn_model, device=self.device, cnn_transform=self.cnn_transform,
            lbp_radius=2, gabor_frequencies=(0.1, 0.3, 0.5), gabor_thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4),
            hist_color_space="HSV", hist_bins=(16, 16, 16), verbose=False
        )

        try: self.segformer_processor = SegformerImageProcessor.from_pretrained("preprocessor_config.json")
        except: self.segformer_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        
        self.det_configs = {"FasterRCNN": {"size": 256}, "RetinaNet": {"size": 512}, "YOLO": {"size": 512}}
        
        # --- UPDATED PATHS using get_path() ---
        self.seg_files = {
            "Attention U-Net": get_path("models/segmentation/AttentionUNet.pth"), 
            "LightweightUNet": get_path("models/segmentation/LightweightUNet.pth"),
            "SegFormer": get_path("models/segmentation/SegFormer.pth"), 
            "DeepLabV3+": get_path("models/segmentation/DeepLabV3Plus.pth")
        }
        self.det_files = {
            "FasterRCNN": get_path("models/detection/FasterRCNN.pt"), 
            "RetinaNet": get_path("models/detection/RetinaNet.pt"), 
            "YOLO": get_path("models/detection/YOLO.pt")
        }
        self.cls_files = {
            "ANN": get_path("models/classification/ANN.pkl"), 
            "KNN": get_path("models/classification/KNN.pkl"), 
            "SVM": get_path("models/classification/SVM.pkl"),
            "ResNet50": get_path("models/classification/ResNet50.pt"), 
            "VGG19": get_path("models/classification/VGG19.pt")
        }

    def _safe_load_weights(self, model, path):
        try: model.load_state_dict(torch.load(path, map_location=self.device))
        except: 
            try: model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False).state_dict())
            except: pass
        return model

    def load_model(self, task, model_name):
        key = f"{task}_{model_name}"
        if key in self.loaded_models: return self.loaded_models[key]
        path = self.seg_files.get(model_name) if task=="segmentation" else (self.det_files.get(model_name) if task=="detection" else self.cls_files.get(model_name))
        if not path or not os.path.exists(path): return None

        if model_name=="YOLO": 
            try: self.loaded_models[key]=YOLO(path); return self.loaded_models[key]
            except: return None
        if str(path).endswith('.pkl'):
            try: self.loaded_models[key]=joblib.load(path); return self.loaded_models[key]
            except: return None

        if model_name=="Attention U-Net": model=smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid", decoder_attention_type="scse"); model=self._safe_load_weights(model, path)
        elif model_name=="LightweightUNet": model=LightweightUNet(pretrained=False); model=self._safe_load_weights(model, path)
        elif model_name=="DeepLabV3+": model=smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation="sigmoid"); model=self._safe_load_weights(model, path)
        elif model_name=="SegFormer": model=SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", num_labels=2, ignore_mismatched_sizes=True); model=self._safe_load_weights(model, path)
        elif model_name=="FasterRCNN": model=fasterrcnn_resnet50_fpn(weights=None); model.roi_heads.box_predictor=FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 8); model=self._safe_load_weights(model, path)
        elif model_name=="RetinaNet": model=retinanet_resnet50_fpn(weights=None, num_classes=8); model=self._safe_load_weights(model, path)
        elif model_name in ["ResNet50", "VGG19"]: 
            try: model=torch.load(path, map_location=self.device, weights_only=False)
            except: return None

        if isinstance(model, nn.Module): model.to(self.device).eval()
        self.loaded_models[key] = model
        return model

    # --- CLASSIFICATION ---
    def _predict_proba_single(self, image, model_name):
        model = self.load_model("classification", model_name)
        if model is None: return None
        if model_name in ["ANN", "KNN", "SVM"]:
            img_np = np.array(image.convert('RGB'))
            features = self.fusion_helper.fuse_features(img_np, use_cnn=True, use_gabor=True, use_lbp=True, use_ch=True).reshape(1, -1)
            if hasattr(model, "predict_proba"): return model.predict_proba(features)[0]
            else: d = model.decision_function(features)[0]; e_x = np.exp(d - np.max(d)); return e_x / e_x.sum()
        else:
            t = T.functional.normalize(T.functional.to_tensor(image.resize((256, 256))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0).to(self.device)
            with torch.no_grad(): return F.softmax(model(t), dim=1).cpu().numpy()[0]

    def predict_fusion_classification(self, image):
        w_sum = np.zeros(7); votes = []
        for name, weight in FUSION_WEIGHTS.items():
            if weight <= 0: continue
            p = self._predict_proba_single(image, name)
            if p is not None:
                idx = np.argmax(p); conf = np.max(p); disease, risk = LESION_INFO[idx]
                votes.append({"model": name, "disease": disease, "confidence": float(conf), "risk": risk})
                w_sum += p * weight
        final_idx = np.argmax(w_sum); final_conf = w_sum[final_idx] / sum(FUSION_WEIGHTS.values())
        return {"final_diagnosis": LESION_INFO[final_idx][0], "risk": LESION_INFO[final_idx][1], "confidence": f"{final_conf:.1%}", "votes": votes}

    # --- DETECTION ---
    def _draw_clean_boxes(self, image, boxes, scores, labels):
        draw = ImageDraw.Draw(image)
        w_orig, h_orig = image.size

        base_scale = max(w_orig, h_orig)

        dynamic_font_size = max(20, int(base_scale * 0.035))

        dynamic_line_width = max(4, int(base_scale * 0.008))

        try:
            font = ImageFont.truetype("arial.ttf", dynamic_font_size)
        except:
            font = ImageFont.load_default()

        if len(boxes) > 0:
            keep = nms(torch.tensor(boxes), torch.tensor(scores), 0.45)
            boxes = boxes[keep.cpu().numpy()]
            scores = scores[keep.cpu().numpy()]
            labels = labels[keep.cpu().numpy()]
            
        for i in np.argsort(scores):
            box = boxes[i]
            score = scores[i]
            idx = int(labels[i])
            
            name = LESION_INFO[idx][0] if idx in LESION_INFO else f"Class {idx}"
            color = BOX_COLORS[idx % len(BOX_COLORS)]
            
            draw.rectangle(box, outline=color, width=dynamic_line_width)
            
            label_text = f"{name} {score:.0%}"
            
            l, t, r, b = draw.textbbox((box[0], box[1]), label_text, font=font)
            
            text_w = r - l
            text_h = b - t
            
            text_x = box[0]
            text_y = box[1] - text_h - 10
            
            if text_y < 0:
                text_y = box[1] + 10 
            
            text_bg_x1 = text_x - 5
            text_bg_y1 = text_y - 5
            text_bg_x2 = text_x + text_w + 5
            text_bg_y2 = text_y + text_h + 5
            
            draw.rectangle((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), fill=color)
            draw.text((text_x, text_y), label_text, fill="white", font=font)
            
        return image

    def predict_detection(self, image, model_name):
        model = self.load_model("detection", model_name)
        if model is None: return image
        w_orig, h_orig = image.size; ts = self.det_configs[model_name]["size"]
        img_draw = image.copy()
        if model_name == "YOLO":
            try:
                res = model(image, imgsz=ts, verbose=False)[0]
                boxes = res.boxes.xyxy.cpu().numpy(); scores = res.boxes.conf.cpu().numpy(); labels = res.boxes.cls.cpu().numpy()
                mask = scores > 0.20
                return self._draw_clean_boxes(img_draw, boxes[mask], scores[mask], labels[mask])
            except: return image
        elif model_name in ["FasterRCNN", "RetinaNet"]:
            t = T.functional.to_tensor(image.resize((ts, ts))).to(self.device).unsqueeze(0)
            with torch.no_grad(): p = model(t)[0]
            scores = p['scores'].cpu().numpy(); mask = scores > 0.20
            if np.sum(mask)==0: return img_draw
            boxes = p['boxes'][mask].cpu().numpy(); labels = p['labels'][mask].cpu().numpy(); scores = scores[mask]
            if model_name == "RetinaNet": boxes = np.clip(boxes, 0, ts)
            sx, sy = w_orig/ts, h_orig/ts; boxes[:, [0, 2]] *= sx; boxes[:, [1, 3]] *= sy
            return self._draw_clean_boxes(img_draw, boxes, scores, labels)
        return img_draw

    # --- SEGMENTATION  ---
    def _predict_seg_raw(self, image, model_name):
        model = self.load_model("segmentation", model_name)
        if model is None: return None
        
        # SegFormer
        if model_name == "SegFormer":
            resized = image.resize((256, 256))
            i = self.segformer_processor(images=resized, return_tensors="pt")
            i = {k: v.to(self.device) for k, v in i.items()}
            with torch.no_grad():
                l = model(**i).logits
                l = F.interpolate(l, size=image.size[::-1], mode="bilinear", align_corners=False)
                p = F.softmax(l, dim=1)[:, 1, :, :].cpu().numpy().squeeze()
        
        # Others
        else:
            n = np.array(image.resize((256, 256)))
            t = torch.from_numpy(n).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad(): 
                out = model(t); out = out['out'] if isinstance(out, dict) else out
                p = out.cpu().numpy().squeeze()
            p = cv2.resize(p, image.size, interpolation=cv2.INTER_LINEAR)
        return p

    def _create_transparent_mask(self, prob_map, color):
        h, w = prob_map.shape
        img = np.zeros((h, w, 4), dtype=np.uint8)
        mask = prob_map > 0.5
        img[mask] = color 
        return Image.fromarray(img, 'RGBA')

    def predict_segmentation_layer(self, image, model_name, color):
        p = self._predict_seg_raw(image, model_name)
        if p is None: return None
        return self._create_transparent_mask(p, color)

    def predict_fusion_segmentation_layer(self, image, color):
        p_seg = self._predict_seg_raw(image, "SegFormer")
        p_att = self._predict_seg_raw(image, "Attention U-Net")
        p_hyb = self._predict_seg_raw(image, "LightweightUNet")
        
        if any(x is None for x in [p_seg, p_att, p_hyb]): return None
        
        fused_prob = (0.60 * p_seg) + (0.20 * p_att) + (0.20 * p_hyb)
        return self._create_transparent_mask(fused_prob, color)

    def predict_multi_segmentation(self, image):
        r = {}
        # 1. Individual Models
        for name in self.seg_files:
            mask = self.predict_segmentation_layer(image, name, SEG_COLORS.get(name, (255,255,255,100)))
            if mask: r[name] = mask
        
        # 2. Fusion
        fusion_mask = self.predict_fusion_segmentation_layer(image, SEG_COLORS["Fusion (Top 3)"])
        if fusion_mask: r["Fusion (Top 3)"] = fusion_mask
        
        return r
    
    def predict_multi_detection(self, image):
        r = {}
        for n in self.det_files: 
            try: r[n] = self.predict_detection(image, n)
            except: r[n] = image
        return r