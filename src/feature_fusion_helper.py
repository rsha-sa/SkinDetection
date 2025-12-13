import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import gabor

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


class FeatureFusionHelperTorch:
    """
    PyTorch-based helper to extract and fuse features:
      - CNN embeddings (torch model)
      - Gabor filters
      - LBP
      - Color Histogram (CH)
    """

    def __init__(
        self,
        cnn_model: nn.Module = None,
        device: torch.device = None,
        cnn_transform: T.Compose = None,
        input_size=(224, 224),
        # LBP params
        lbp_radius=2,
        lbp_n_points=None,
        lbp_method="uniform",
        lbp_n_bins=None,
        # Gabor params
        gabor_frequencies=(0.1, 0.3, 0.5),
        gabor_thetas=(0, np.pi/4, np.pi/2, 3*np.pi/4),
        # CH params
        hist_color_space="HSV",
        hist_bins=(16, 16, 16),
        hist_range=(0, 256),
        verbose=False
    ):
        """
        verbose: print detailed info during processing
        """
        self.verbose = verbose
        self.cnn_model = cnn_model

        # Device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.input_size = input_size

        # CNN model setup
        if cnn_model is not None:
            self.cnn_model.to(self.device)
            self.cnn_model.eval()

        # CNN transform
        if cnn_transform is None:
            self.cnn_transform = T.Compose([
                T.Resize(input_size),
                T.ToTensor(),
            ])
        else:
            self.cnn_transform = cnn_transform

        # LBP parameters
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points if lbp_n_points is not None else 8 * lbp_radius
        self.lbp_method = lbp_method
        self.lbp_n_bins = lbp_n_bins if lbp_n_bins is not None else self.lbp_n_points + 2

        # Gabor params
        self.gabor_frequencies = gabor_frequencies
        self.gabor_thetas = gabor_thetas

        # Color hist params
        self.hist_color_space = hist_color_space.upper()
        self.hist_bins = hist_bins
        self.hist_range = hist_range

        if self.verbose:
            print(f"[FusionHelper] Initialized with verbose={self.verbose}")
            print(f"[FusionHelper] Input resize: {self.input_size}")
            print(f"[FusionHelper] LBP: P={self.lbp_n_points}, R={self.lbp_radius}")
            print(f"[FusionHelper] Gabor frequencies={self.gabor_frequencies}, thetas={self.gabor_thetas}")
            print(f"[FusionHelper] Color hist={self.hist_color_space}, bins={self.hist_bins}")


    # -----------------------------------------------------
    # CNN FEATURES
    # -----------------------------------------------------
    def extract_cnn_features(self, img_rgb):
        if self.cnn_model is None:
            return None

        if self.verbose:
            print("[FusionHelper] Extracting CNN features...")

        pil_img = Image.fromarray(img_rgb)
        x = self.cnn_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.cnn_model(x)

        return feats.cpu().numpy().squeeze().astype("float32")


    # -----------------------------------------------------
    # LBP FEATURES
    # -----------------------------------------------------
    def extract_lbp_features(self, img_rgb):
        if self.verbose:
            print("[FusionHelper] Extracting LBP features...")

        gray = rgb2gray(img_rgb)
        lbp = local_binary_pattern(
            gray,
            P=self.lbp_n_points,
            R=self.lbp_radius,
            method=self.lbp_method
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.lbp_n_bins,
            range=(0, self.lbp_n_bins),
            density=True
        )
        return hist.astype("float32")


    # -----------------------------------------------------
    # GABOR FEATURES
    # -----------------------------------------------------
    def extract_gabor_features(self, img_rgb):
        if self.verbose:
            print("[FusionHelper] Extracting Gabor features...")

        gray = rgb2gray(img_rgb)
        feats = []

        for freq in self.gabor_frequencies:
            for theta in self.gabor_thetas:
                real, imag = gabor(gray, frequency=freq, theta=theta)
                magnitude = np.sqrt(real ** 2 + imag ** 2)
                feats.append(magnitude.mean())
                feats.append(magnitude.std())

        return np.array(feats, dtype="float32")


    # -----------------------------------------------------
    # COLOR HISTOGRAM FEATURES
    # -----------------------------------------------------
    def extract_color_hist_features(self, img_rgb):
        if self.verbose:
            print("[FusionHelper] Extracting color histogram...")

        img = img_rgb.copy()

        if self.hist_color_space == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        hist = cv2.calcHist(
            [img],
            [0, 1, 2],
            None,
            self.hist_bins,
            self.hist_range * 3
        )

        hist = hist.flatten()
        hist /= (hist.sum() + 1e-8)

        return hist.astype("float32")


    # -----------------------------------------------------
    # ALL FEATURES + FUSION
    # -----------------------------------------------------
    def extract_all_features(self, img_rgb):
        if self.verbose:
            print("[FusionHelper] Resizing and extracting all features...")

        img_resized = cv2.resize(img_rgb, self.input_size[::-1])

        return {
            "cnn": self.extract_cnn_features(img_resized),
            "gabor": self.extract_gabor_features(img_resized),
            "lbp": self.extract_lbp_features(img_resized),
            "ch": self.extract_color_hist_features(img_resized),
        }


    def fuse_features(self, img_rgb, use_cnn=True, use_gabor=True, use_lbp=True, use_ch=True):
        if self.verbose:
            print("[FusionHelper] Fusing features...")

        print(use_cnn, use_gabor, use_lbp, use_ch)
        feats = self.extract_all_features(img_rgb)
        parts = []

        if use_cnn and feats["cnn"] is not None:
            parts.append(feats["cnn"])
        if use_gabor:
            parts.append(feats["gabor"])
        if use_lbp:
            parts.append(feats["lbp"])
        if use_ch:
            parts.append(feats["ch"])

        return np.concatenate(parts).astype("float32")


    # -----------------------------------------------------
    # DATASET-LEVEL EXTRACT
    # -----------------------------------------------------
    def transform_dataset(self, images, **fuse_kwargs):
        if self.verbose:
            print(f"[FusionHelper] Transforming dataset of {len(images)} images...")

        all_feats = []
        for idx, img in enumerate(images):
            if self.verbose:
                print(f"  -> Processing image {idx+1}/{len(images)}...")
            all_feats.append(self.fuse_features(img, **fuse_kwargs))

        return np.vstack(all_feats)
