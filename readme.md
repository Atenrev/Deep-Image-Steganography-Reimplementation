# Deep-Image-Steganography-Reimplementation
## Description
This is a PyTorch Reimplementation of [High-Capacity Convolutional Video Steganography with Temporal Residual Modeling (ICMR2019, oral)](https://dl.acm.org/doi/abs/10.1145/3323873.3325011).

* The pre-trained model provided was trained on the tiny_imagenet dataset.


## Prerequisites
* PyTorch
* torchvision
* tqdm


## Installation
### 1. Clone the repo

```
git clone https://github.com/Atenrev/Deep-Image-Steganography-Reimplementation.git
cd Deep-Image-Steganography-Reimplementation
```

### 2. Data
* Download Train, Validation and Test Data: [Tiny-Imagenet-200](http://cs231n.stanford.edu/tiny-imagenet-200.zip). Put it in the root folder.


Here is an example:
```
.
├── dataset.py
├── model.py
├── inference.py
├── trainer.py
├── model
└── tiny_imagenet
    ├── train
    ├── val
    └── test
```

You can also use your own dataset. In that case, pass the ```--train_dataset``` and ```--val_dataset``` to the ```trainer.py``` arguments.

## Train
Run ```train.py```:

``` sh
python train.py 

optional arguments:
  -h, --help                        show this help message and exit
  --train_dataset TRAIN_DATASET     location of the dataset
  --val_dataset VAL_DATASET         location of the dataset
  --batch_size BATCH_SIZE           training batch size
  --image_size IMAGE_SIZE           image size
  --val_perc VAL_PERC               validation data percentage
  --epochs EPOCHS                   number of training epochs
  --lr LR                           learning rate
  --beta1 BETA1                     adam beta
  --save_dir SAVE_DIR               save model directory
  --load_model LOAD_MODEL           model to load and resume training
```
## Inference (single sample)
Run ```inference.py```:

``` sh
python .\inference.py --mode r --im "merged.jpg"
```