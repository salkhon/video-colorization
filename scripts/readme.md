# Dataset installation

## DAVIS
```
wget https://cgl.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
unzip -q ./DAVIS-data.zip
rm ./DAVIS-data.zip
rm -rf ./DAVIS/JPEGImages/1080p
```

## VQA v2
```
wget http://images.cocodataset.org/zips/val2014.zip
unzip -q ./val2014.zip
rm ./val2014.zip
```

# Training scripts
Training (no pruning/quantization):
```
python train.py
```

Training with pruning (after normal training):
```
python train_with_pruning.py
```
Quantize pruned model (after training with pruning):
```
python train_with_pruning_quantize.py
```