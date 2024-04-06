# Locatioanl Persistence Images: Wafer Map Pattern Recognition Based on TDA

**2023 KSIAM Annual Meeting Poster**

## Index
- [Abstract](#Abstract)
- [Introduction](#Introduction)
- [Method](#Method)
- [Experiment](#Experiment)
- [**Usage**](#Usage)

## Abstract
Wafer mapping, as it provides crucial information for identifying the root causes of defects, is an essential step in semiconductor manufacturing. Recently, a new approach was proposed for the wafer defect pattern classification using Topological Data Analysis (TDA). Throughout various experiments, this method has shown superior performances compared to the conventioanl CNN-based methods. However, ti soely focused on the shape of defect and cannot recognize the locational information of defects. To address this issues, we propose the Locational Persistence Images (LPI), which accurately and efficiently classfy both the shape and location of defects patterns simultaneously. Through several experiments on simulated datasets, we confirmed that this proposed method is faster and achieves competitive accuracy compared to the existing methods.

## Introduction
A wafer map enables engineers to maintain product quality and optain better yields. In the past, specialized engineers manually inspected each wafer for wafer testing, which was both time-consuming and costly, heavily depend on engineer expertise. For automation, recently, deep learning has been utilized for the classification of defect patterns in wafer maps.

### 1) Method based on CNN
<p align="center"> <img src=https://github.com/larpp/WMDPR-based-on-TDA/assets/87048326/55b1e122-5a67-49fa-ba56-371779eb43e9 width="70%" height="70%">

### 2) Method based on TDA
<p align="center"> <img src=https://github.com/larpp/WMDPR-based-on-TDA/assets/87048326/88a3510c-77de-4d9d-927a-acf7e385ca0d width="70%" height="70%">

## Method
The existing method using TDA can classify only the shape of defects and cannot classify the location of defects. Therefore, we propose an improved approach to classify wafer map defect patterns using LPI (Locational Persistence Images) generated by directly applying a Gaussian distribution to the wafer map.

### New Method based on LPI
Our new approach adds another PI to the existing methods. This PI is created directly by imposing the Gaussian functions on each defect. Then we concatenate a total of three PIs, including the 0-dim PI, 1-dim PI, and LPI, to use as input a single 300-dimensional vector. Trough this approach, we can classify 22 classes of wafer map, which also concerns locational information.
![스크린샷 2024-04-06 183326](https://github.com/larpp/WMDPR-based-on-TDA/assets/87048326/dbe58b77-af13-413f-aee7-955a0c4045d7)

## Experiment

### 1) Training and validation accuracy
![스크린샷 2024-04-06 142252](https://github.com/larpp/WMDPR-based-on-TDA/assets/87048326/d8eb2f31-6070-4b8d-b00c-8ee10afb80d4)

|  | CNN | TDA | LPI |
|:---------:|:----------:|:---------:|:--------:|
| train | 100 | 47.68 | 100 |
| test | 95.77 | 42.95 | 96.09 |

### 2) Small-data Experiment
Test accuracy (%) for two models trained with a small amount of training data
| #Training data | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LPI | 87.27 | 94.09 | 94.85 | 95.91 | 94.73 | 94.39 | 95.13 | 95.17 | 96.29 | 95.39 |
| CNN | 66.82 | 80.68 | 83.48 | 85.57 | 86.45 | 86.74 | 90.84 | 91.65 | 91.26 | 91.68 |

### 3) Imbalanced-data Experiment
Number of training data & Test accuracy (%) for two models trained with imbalanced datasets
|  | C1 | C2 | C3 | C4 | C5 | C6 | C7 | C8 | C9 | C10 | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | C20 | C21 | C22 | **LPI** | **CNN** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Dataset1 | 69 | 292 | 33 | 131 | 61 | 254 | 231 | 242 | 195 | 108 | 49| 250 | 15 | 200 | 222 | 2 | 229 | 137 | 118 | 53 | 163 | 16 | **94.59** | **83.32** |
| Dataset2 | 40 | 205 | 21 | 118 | 265 | 78 | 299 | 276 | 17 | 16 | 36 | 164 | 28 | 16 | 70 | 281 | 152 | 271 | 169 | 48 | 204 | 246 | **94.68** | **85.82** |
| Dataset3 | 156 | 55 | 294 | 54 | 125 | 43 | 157 | 150 | 19 | 197 | 202 | 194 | 218 | 278 | 150 | 199 | 110 | 285 | 114 | 82 | 16 | 147 | **95.36** | **89.32** |
| Dataset4 | 226 | 102 | 228 | 263 | 143 | 264 | 43 | 263 | 296 | 1 | 69 | 114 | 100 | 19 | 20 | 236 | 204 | 235 | 68 | 18 | 22 | 183 | **90.27** | **83.91** |
| Dataset5 | 30 | 248 | 251 | 51 | 133 | 214 | 220 | 259 | 8 | 37 | 108 | 25 | 146 | 30 | 62 | 100 | 297 | 47 | 88 | 251 | 26 | 201 | **94.18** | **86.18** |

### 4) Training Efficieny
| | Training time |
|:---------:|:----------:|
| LPI | 0.239 s/epoch |
| CNN | 17.2 s/epoch |

## Usage
```
pip install -r requirements.txt
```
### LPI
**Generate LPI datasets**
```
python PI_data.py
```

**Train**
```
python main.py
```

**Test**
```
python main.py --test
```

#### Small & Imbalanced data experiments
**Generate small or imbalanced dataset**
```
# Small datasets
sh tools/make_small_pi.sh

# Imbalanced datasets
sh tools/make_imbalance_pi.sh
```

**Train**
```
# Small datasets
sh tools/train_small_lpi.sh

# Imbalanced datasets
sh tools/train_imbalance_lpi.sh
```

**Test**
```
# Small datasets
sh tools/test_small_lpi.sh

# Imbalanced datasets
sh tools/test_imbalance_lpi.sh
```

### CNN
**Generate image datasets**
```
python img_data.py
```

**Train with single GPU**
```
python resnet50.py
```

**Train with multiple GPU**
```
python main.py --num_workers 8 --multi_gpu
```

**Test**
```
python resnet50.py --test
```

#### Small & Imbalanced data experiments
**Generate small or imbalanced dataset**
```
# Small datasets
sh tools/make_small_img.sh

# Imbalanced datasets
sh tools/make_imbalance_img.sh
```

**Train**
```
# Small datasets
sh tools/train_small_resnet50.sh

# Imbalanced datasets
sh tools/train_imbalance_resnet50.sh
```

**Test**
```
# Small datasets
sh tools/test_small_resnet50.sh

# Imbalanced datasets
sh tools/test_imbalance_resnet50.sh
```

### TDA
**Generate TDA datasets**
```
python PI_data.py --path TDA_data --lpi
```

**Train**
```
python main.py --pi_path TDA_data --save_model results_tda.pt --input_size 200 --save_acc_name TDA
```

**Test**
```
python main.py --pi_path TDA_data --save_model results_tda.pt --input_size 200 --test
```
