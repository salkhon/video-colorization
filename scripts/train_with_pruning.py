import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, random_split, Subset
from torchvision import transforms
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
import torchvision
import torchvision.models as models
from torch.optim import AdamW
from torch.nn import MSELoss
from sklearn.model_selection import train_test_split
from kornia import color
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import os
import matplotlib.pyplot as plt
import copy
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class datasetTrain(Dataset):
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

PERCEPTUAL_LOSS_WEIGHT = 0.3
def train(model, vgg11, train_loader, val_loader, loss1, loss2, optimizer, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in tqdm(train_loader):      
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            
            loss = loss1(output, y) + PERCEPTUAL_LOSS_WEIGHT*loss2(vgg11(output),vgg11(y))
        
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
#         train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        
        if val_loader is None:
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")
            continue
            
        with torch.no_grad():
            for x, y in tqdm(val_loader):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = loss1(output, y) + loss2(vgg11(output),vgg11(y))
                val_loss += loss.item()
#             val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

def get_all_layers(model):
    layers = []

    def _get_layers_recursive(module):
        if not list(module.children()):
            layers.append(module)
        else:
            for child in module.children():
                _get_layers_recursive(child)
    _get_layers_recursive(model)
    return layers

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements

def get_next_conv_layer(layers, k):
    for i in range(k+1,len(layers)):
        if isinstance(layers[i], nn.Conv2d) and layers[k].weight.shape[1] == layers[i].weight.shape[0]:
            
            return i
    return -1

def get_prev_conv_layer(layers, k):
    for i in reversed(range(0,k)):
        if isinstance(layers[i], nn.Conv2d):
            return i
    return -1

def get_num_channels_to_keep(channels: int, prune_ratio: float) -> int:
    return int(round(channels*(1-prune_ratio)))

def find_index(layers, target):
    for i, layer in enumerate(layers):
        if layer == target:
            return i
        
def get_out_channel_importance(weight):
    out_channels = weight.shape[0]
    importances = []
    # compute the importance for each input channel
    for o_c in range(weight.shape[0]):
        channel_weight = weight.detach()[o_c]
        importance = torch.linalg.norm(channel_weight)
        importances.append(importance.view(1))
    return torch.cat(importances)

def get_keep_indices(conv_layer, p_ratio):
    n_keep = get_num_channels_to_keep(conv_layer.out_channels, p_ratio)
    importance = get_out_channel_importance(conv_layer.weight)
    sort_idx = torch.argsort(importance, descending=True)
    return sort_idx[:n_keep]

def main(directory, vid_directory):
    # directory = "./val2014"
    # vid_directory = './DAVIS/JPEGImages/480p'

    for folder in os.listdir(vid_directory):
        path = os.path.join(vid_directory, folder)
        resize_images(path)

    dataset = datasetTrain(Path(directory))
    vid_dataset = datasetTrain(Path(vid_directory))

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

    probabilities = (occurrence/torch.sum(occurrence)).view(-1)
    probabilities = -torch.log(probabilities+1e-6).view(-1,1)

    value_embedding = nn.Embedding(num_embeddings=NUM_BINS, embedding_dim=1) 
    value_embedding.weight.data = probabilities
    value_embedding = value_embedding.to("cuda")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNetWithEfficientNetFeatureExtractor()

    vgg11 = models.vgg11(pretrained=True)
    vgg11.features[-1]=torch.nn.Identity()
    vgg11 = vgg11.features

    for param in vgg11.parameters():
        param.requires_grad = False
        
    vgg11 = vgg11.to(device)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    vid_loader = DataLoader(vid_dataset, batch_size=64, shuffle=True)

    model.load_state_dict(torch.load("./model.pth"))
    model2 = copy.deepcopy(model)

    model2 = model2.to("cpu")
    model2.eval()
    layers = get_all_layers(model2.feature_extractor)

    prune_type_1 = []
    prune_type_2 = []
    prune_type_3 = []
    prune_type_4 = []

    for i in range(len(layers)-2):
        if isinstance(layers[i], nn.Conv2d) and isinstance(layers[i+1], nn.BatchNorm2d):
            if isinstance(layers[i+2], nn.SiLU):
                prune_type_1.append(i)
            else:
                prune_type_2.append(i)
        if isinstance(layers[i], nn.Conv2d) and isinstance(layers[i+1],nn.Conv2d):
            prune_type_3.append(i)
            prune_type_4.append(i+1)

    conv_layer = model2.colorization_layers[1].conv
    bn_layer =  model2.colorization_layers[1].bn
    next_conv =  model2.colorization_layers[2].conv

    p_ratio = 0.3
    n_keep = get_num_channels_to_keep(conv_layer.out_channels, p_ratio)
    keep_idx = get_keep_indices(conv_layer, p_ratio)

    with torch.no_grad():
        bn_layer.weight.set_(bn_layer.weight.detach()[keep_idx])
        bn_layer.bias.set_(bn_layer.bias.detach()[keep_idx])
        bn_layer.running_mean.set_(bn_layer.running_mean.detach()[keep_idx])
        bn_layer.running_var.set_(bn_layer.running_var.detach()[keep_idx])

        conv_layer.out_channels = n_keep
        conv_layer.weight.set_(conv_layer.weight.detach()[keep_idx])

        if conv_layer.bias is not None:
            conv_layer.bias.set_(conv_layer.bias.detach()[keep_idx])

        if next_conv.groups > 1:
            next_conv.groups = n_keep

        next_conv.in_channels = n_keep
        next_conv.weight.set_(next_conv.weight.detach()[:,keep_idx,:,:])

    model2.feature_extractor[8] = nn.Identity()
    model2.colorization_layers[0] = nn.Identity()

    p_ratio = 0.5
    for i in prune_type_3:
        conv_layer = layers[i]    
        next_conv_layer = layers[i+1]

        n_keep = get_num_channels_to_keep(conv_layer.out_channels, p_ratio)
        keep_idx = get_keep_indices(conv_layer, p_ratio)
        
        conv_layer.out_channels = n_keep
        conv_layer.weight.set_(conv_layer.weight.detach()[keep_idx])
        
        if conv_layer.bias is not None:
            conv_layer.bias.set_(conv_layer.bias.detach()[keep_idx])

        next_conv_layer.weight.set_(next_conv_layer.weight.detach()[:,keep_idx,:,:])
        if next_conv_layer.groups > 1:
            next_conv_layer.groups = n_keep
        next_conv_layer.in_channels = n_keep

    model2.to(device)
    model2.requires_grad_= True
    optimizer = AdamW(model2.parameters(), lr=1e-5)
    mseloss = MSELoss()
    weighted_mseloss = WeightedMSELoss(value_embedding)

    train(model2, vgg11, train_loader, val_loader, weighted_mseloss, mseloss, optimizer, device, epochs=1)

    torch.save(model2.state_dict(), "model_p.pth")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training model.")
    parser.add_argument("--training_dataset_directory", type=str, help="Dataset directory for training (default = ./val2014)", default="./val2014")
    parser.add_argument("--sample_dataset_directory", type=str, help="Dataset directory for sample inference (default = ./DAVIS/JPEGImages/480p)", default="./DAVIS/JPEGImages/480p")
    # parser.add_argument("--epochs", type=int, help="Number of epochs (default = 2)", default=2)
    args = parser.parse_args()

    main(args.training_dataset_directory, args.sample_dataset_directory)
