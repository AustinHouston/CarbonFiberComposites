import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from vk4_reader import vk4extract

from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage import filters, morphology, feature, measure
from scipy.ndimage import binary_dilation, binary_closing
from skimage.morphology import disk
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from collections import Counter


# Dataset
# --------------------------------------------------------------------------------------------------
class SegmentationDataset(Dataset):
    def __init__(self, image, label, patch_size=(128, 128), transform=None):
        self.image = image
        self.label = label
        self.patch_size = patch_size
        self.transform = transform
        self.patches = self._create_patches()

    def _create_patches(self):
        h, w = self.image.shape
        ph, pw = self.patch_size
        patches = []

        for i in range(0, h - ph + 1, ph):
            for j in range(0, w - pw + 1, pw):
                image_patch = self.image[i:i+ph, j:j+pw]
                label_patch = self.label[i:i+ph, j:j+pw]
                patches.append((image_patch, label_patch))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image_patch, label_patch = self.patches[idx]
        image_patch = torch.from_numpy(image_patch).float().unsqueeze(0)  # Add channel dimension (1, H, W)
        label_patch = torch.from_numpy(label_patch).long()  # For segmentation, labels should be long
        
        if self.transform:
            image_patch = self.transform(image_patch)
        return image_patch, label_patch


# Class Weights
# --------------------------------------------------------------------------------------------------
def compute_class_weights(label_mask):
    # Flatten the label mask and count occurrences of each class
    class_counts = Counter(label_mask.flatten())
    total_pixels = sum(class_counts.values())

    # Compute weights: inverse frequency
    class_weights = {cls: total_pixels / count for cls, count in class_counts.items()}
    
    # Normalize weights to sum to 1
    total_weight = sum(class_weights.values())
    class_weights = {cls: weight / total_weight for cls, weight in class_weights.items()}
    
    return [class_weights.get(cls, 1.0) for cls in range(len(class_counts))]

# Neaural Network
# --------------------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        conv = self.conv_block(x)
        x_pooled = self.pool(conv)
        return conv, x_pooled

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class Unet(nn.Module):
    def __init__(self, input_channels, num_classes, num_filters, dropout = 0.1):
        super(Unet, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # Encoder
        for i, filters in enumerate(num_filters[:-1]):
            self.encoders.append(EncoderBlock(input_channels if i == 0 else num_filters[i-1], filters))

        # Bottleneck
        self.bottleneck = ConvBlock(num_filters[-2], num_filters[-1])

        # Decoder
        num_filters_reversed = num_filters[::-1]
        for i in range(len(num_filters) - 1):
            in_channels = num_filters_reversed[i]
            out_channels = num_filters_reversed[i + 1]  # This is the output channel size after ConvBlock
            self.decoders.append(DecoderBlock(in_channels, out_channels))

        # Classifier
        self.final_conv = nn.Conv2d(num_filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x, x_pooled = encoder(x)
            skips.append(x)
            x = x_pooled

        x = self.bottleneck(x)
        x = self.dropout(x)

        # reverse the skips list
        skips_reverse = skips[::-1]

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips_reverse[i])

        x = self.final_conv(x)

        return x


# Get the number of trainable parameters
# --------------------------------------------------------------------------------------------------
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Predictions
# --------------------------------------------------------------------------------------------------
def predict(model, image_patch, device):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(image_patch.to(device))  # Forward pass
        prediction = torch.sigmoid(output).cpu().squeeze(0).squeeze(0)  # Convert to 1xHxW
    return prediction


def predict_full_image(model, device, image, patch_size=40, stride=20, edge_crop=3):
    model.eval()
    height, width = image.shape
    prediction = np.zeros((height, width))
    counts = np.zeros((height, width))  # To keep track of overlapping predictions for averaging
    
    with torch.no_grad():
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # Extract patch
                patch = image[y:y + patch_size, x:x + patch_size]
                patch = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions
                
                # Predict
                pred = torch.sigmoid(model(patch)).cpu().squeeze(0).squeeze(0).numpy()[1]
                pred = pred[edge_crop:-edge_crop, edge_crop:-edge_crop]
                # Accumulate prediction
                prediction[y+edge_crop:y + patch_size-edge_crop, x+edge_crop:x + patch_size-edge_crop] += pred
                counts[y+edge_crop:y + patch_size-edge_crop, x+edge_crop:x + patch_size-edge_crop] += 1
    
    # Avoid division by zero and normalize by the number of overlapping patches
    prediction /= np.maximum(counts, 1)
    return prediction