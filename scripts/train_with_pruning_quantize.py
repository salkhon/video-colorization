import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import os
import matplotlib.pyplot as plt

INTERVAL = 10
INTERVAL_SQUARED = INTERVAL**2
NUM_BINS = INTERVAL**3+INTERVAL**2+INTERVAL+1


def convert_rgb_tensor_to_key(tensor_data):
    scaled_data = (tensor_data*INTERVAL).to(torch.long)
    bucket = scaled_data[:,0]*INTERVAL_SQUARED+scaled_data[:,1]*INTERVAL+scaled_data[:,2]
    return bucket

class WeightedMSELoss(nn.MSELoss):
    
    def __init__(self, value_embedding_layer):
        super().__init__(reduction="none")
        self.value_embedding_layer = value_embedding_layer

        
    def forward(self, input, target):
        mse_loss = super().forward(input, target)
        bucket = convert_rgb_tensor_to_key(target)
        value_embeddings = self.value_embedding_layer(bucket).squeeze()
        mse_loss = torch.mean(mse_loss, dim=1)
        weighted_mse_loss = (mse_loss * value_embeddings).mean()
        return weighted_mse_loss

import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
# from torchsummary import summary
from kornia import color

class Places365Train(Dataset):
    def __init__(self, root: Path):
        self.data_dir = root
        self.data_paths = list(self.data_dir.rglob("*.jpg"))

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read an image and convert it to black and white and lab color space.

        Args:
            idx (int): Image index

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Black and white image and lab color space image
        """
        with Image.open(self.data_paths[idx]) as img:
            rgb_img = transforms.ToTensor()(img.convert("RGB"))  # (3, 256, 256)
            bw_img = transforms.ToTensor()(img.convert("L"))  # (1, 256, 256)
            bw_img = bw_img.repeat(3, 1, 1)  # (3, 256, 256)

        lab_img = color.rgb_to_lab(rgb_img)  # (3, 256, 256) -> L, a, b
        lab_img[0] = lab_img[0] / 100  # L channel
        lab_img[1:] = lab_img[1:] / 128  # a, b channel

        return bw_img, rgb_img

    @staticmethod
    def lab_to_rgb(lab_img: torch.Tensor) -> torch.Tensor:
        return lab_img
        """Convert lab image to rgb image.

        Args:
            lab_img (torch.Tensor): Lab color space image

        Returns:
            torch.Tensor: RGB image
        """
        lab_img[0] = lab_img[0] * 100  # L channel
        lab_img[1:] = lab_img[1:] * 128  # a, b channel
        rgb_img = color.lab_to_rgb(lab_img)
        return rgb_img
    
def resize_images(directory):
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Or other image extensions
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            if img.width == 256 and img.height == 256:
                break
            img = img.resize((256, 256), Image.LANCZOS)  # Resize with high-quality resampling
            img.save(img_path, quality=95)  

directory = "./val2014"
vid_directory = './DAVIS/JPEGImages/480p'

for folder in os.listdir(vid_directory):
    path = os.path.join(vid_directory, folder)
    resize_images(path)

dataset = Places365Train(Path(directory))
vid_dataset = Places365Train(Path(vid_directory))

import torch
from torch.utils.data import random_split

train_ratio = 0.8
test_ratio = 0.2  
dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
test_size = dataset_size - train_size

train_set, val_set = random_split(dataset, [train_size, test_size])

occurrence = torch.zeros(NUM_BINS)
for data in tqdm(vid_dataset):
    rgb_image = data[1]
    key = convert_rgb_tensor_to_key(rgb_image).view(-1)
    bin_counts = torch.bincount(key, minlength = NUM_BINS)
    occurrence += bin_counts

import matplotlib.pyplot as plt

probabilities = (occurrence/torch.sum(occurrence)).view(-1)
probabilities = -torch.log(probabilities+1e-6).view(-1,1)

value_embedding = nn.Embedding(num_embeddings=NUM_BINS, embedding_dim=1) 
value_embedding.weight.data = probabilities
value_embedding = value_embedding.to("cuda")

class Block(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.Tanh()  # todo: change to ReLU
        self.upsample = torch.nn.Upsample(scale_factor=upsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(self.relu(self.bn(self.conv(x))))

class ConvNetWithEfficientNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.FEATURE_EXTRACTOR_LAYERS = 9
        self.feature_extractor = efficientnet_b7(
            weights=EfficientNet_B7_Weights.IMAGENET1K_V1
        ).features
        self.feature_extractor.requires_grad_(False)

        self.colorization_layers = torch.nn.Sequential(
            Block(2560, 640),
            Block(640, 384),
            Block(384, 224, upsample=2),
            Block(224, 160),
            Block(160, 80, upsample=2),
            Block(80, 48, upsample=2),
            Block(48, 32, upsample=2),
            Block(32, 64, upsample=2),
            torch.nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid(),
        )
        self.quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature32 = self.feature_extractor[0:2](
            x
        )  # 3x256x256 -> 64x128x128 -> 32x128x128
        feature48 = self.feature_extractor[2](feature32)  # 32x128x128 -> 48x64x64
        feature80 = self.feature_extractor[3](feature48)  # 48x64x64 -> 80x32x32
        feature224 = self.feature_extractor[4:6](
            feature80
        )  # 80x32x32 -> 160x16x16 -> 224x16x16
        feature640 = self.feature_extractor[6:8](
            feature224
        )  # 224x16x16 -> 384x8x8 -> 640x8x8
        feature2560 = self.feature_extractor[8](feature640)  # 640x8x8 -> 2560x8x8
        
        if not self.quantized:
            out = self.colorization_layers[0](feature2560)  # 2560x8x8 -> 640x16x16
            out += feature640
            out = self.colorization_layers[1:3](out)  # 640x16x16 -> 384x32x32 -> 224x32x32
            out += feature224
            out = self.colorization_layers[3:5](out)  # 224x32x32 -> 160x64x64 -> 80x64x64
            out += feature80
            out = self.colorization_layers[5](out)  # 80x64x64 -> 48x128x128
            out += feature48
            out = self.colorization_layers[6](out)  # 48x128x128 -> 32x256x256
            out += feature32
            out = self.colorization_layers[7:10](
                out
            )  # 32x256x256 -> 64x256x256 -> 2x256x256
        else:
            out = self.colorization_layers[0](feature2560)  # 2560x8x8 -> 640x16x16
            out += feature640
            out = self.colorization_layers[1](out)  # 640x16x16 -> 384x32x32 -> 224x32x32
            out += feature224
            out = self.colorization_layers[2](out)  # 224x32x32 -> 160x64x64 -> 80x64x64
            out += feature80
            out = self.colorization_layers[3](out)  # 80x64x64 -> 48x128x128
            out += feature48
            out = self.colorization_layers[4](out)  # 48x128x128 -> 32x256x256
            out += feature32
            out = self.colorization_layers[5](out)  # 32x256x256 -> 64x256x256 -> 2x256x256
        return out

    def convert_output_to_rgb(
        self, x: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:

        return output
        lab_img = x.clone()
        lab_img[1:] = output
        return Places365Train.lab_to_rgb(lab_img)
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNetWithEfficientNetFeatureExtractor().to(device)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
vid_loader = DataLoader(vid_dataset, batch_size=64, shuffle=True)

import copy
model = ConvNetWithEfficientNetFeatureExtractor()
model.load_state_dict(torch.load("./model_p.pth"))
model2 = copy.deepcopy(model)
model2 = model2.to("cpu")
val_loader = DataLoader(val_set, batch_size=256, shuffle=True)

feature32 = model2.feature_extractor[0:2](x) 
feature48 = model2.feature_extractor[2](feature32) 
feature80 = model2.feature_extractor[3](feature48) 
feature224 = model2.feature_extractor[4:6](feature80)
feature640 = model2.feature_extractor[6:8](feature224)
feature2560 = model2.feature_extractor[8](feature640)

import torch
from torch import nn
import copy

backend = "fbgemm"
m = copy.deepcopy(model2).to("cpu")
m.eval()
ITERATIONS = 5

def get_quantized(layer, sample_input):
    q = nn.Sequential(torch.quantization.QuantStub(), 
                  layer, 
                  torch.quantization.DeQuantStub())
    q.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(q, inplace=True)

    with torch.inference_mode():
        for _ in range(ITERATIONS):
            out = q(sample_input)
    torch.quantization.convert(q, inplace=True)
    return q, out

q0, out = get_quantized(m.colorization_layers[0],feature2560)
q1, out = get_quantized(m.colorization_layers[1:3],feature640+out)
q2, out = get_quantized(m.colorization_layers[3:5],feature224+out)
q3, out = get_quantized(m.colorization_layers[5],feature80+out)
q4, out = get_quantized(m.colorization_layers[6],feature48+out)
q5, out = get_quantized(m.colorization_layers[7:10],feature32+out)

val_loader = DataLoader(val_set, batch_size=1000)

model2.colorization_layers = nn.Sequential(q0,q1,q2,q3,q4,q5)
model2.quantized = True

torch.save(model2.state_dict(), "model_pq.pth")
torch.cuda.empty_cache()