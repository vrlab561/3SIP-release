# Three-Sensor Inertial Poser (3SIP): Full-Body Motion Generation Using Only Three Inertial Measurement Units

Abstract
---
> Existing 3-D full-body human motion generation methods with inertial measurements often use more than three inertial measurement units (IMUs). This still makes it impractical to widely apply these approaches for personal gaming and virtual reality (VR) applications due to their intrusiveness and hardware costs. In this paper, we present a novel multi-stage approach to generating 3-D full-body human motion sequences with global translation using only three IMUs worn on the head and wrists. Our proposed method first estimates the full-body pose sequence through the Temporal Pose Estimation module. This process is aided by the global velocities of the head, wrists, and pelvis estimated through the Global Velocity Estimation module. The proposed method then refines the pose sequence with the guidance of the Vector Quantized Variational Autoencoder (VQ-VAE) module and adopts a fusion strategy to generate the final motion sequence. The global translation of the pose sequence is estimated by integrating the pelvis (root) velocity. Our method was evaluated on the AMASS dataset, and the IMU motion capture (MoCap) datasets: Xsens and TotalCapture. Experimental results show that our approach outperforms the state-of-the-art (SOTA) methods using the three IMUs with the same setup. 

## Enviroment Setup
All experiments are done on a single Nvidia Geforce RTX 3090 GPU with 24G graphics memory. The code was tested with PyTorch 1.11.0 using Python 3.8.

To create the environment, run the following commands:
```
conda env create -f environment.yml
conda activate 3SIP
```

Download the [human_body_prior](https://github.com/nghorbani/human_body_prior/tree/master/src) lib and put them in this repo. The repo should look like
```
3SIP
├── human_body_prior/
├──── body_model/
├──── data/
├──── ...
├── dataset/
├── prepare_data/
└── ...
```

**Optional:** To save GPU memory, we omitted unnecessary mesh computations. Specifically, we removed the verts variable and its associated operations in the lbs() function of lbs.py (from lines 239 to 252), and also removed the verts variable in the forward() function of body_model.py (lines 257, 264, and 267).

**Note:** 32GB or more GPU memory may be needed if the the computations about verts of SMPL models are kept when training 3SIP on AMASS datasets.

## Dataset Preparation

### SMPL models
We used the smpl model, download it from [here](https://mano.is.tue.mpg.de/) (SMPL-H) and [here](https://smpl.is.tue.mpg.de/) (DMPLS and SMPL)  and place in  `./body_models/`.  Both .npz and .pkl format are needed.

The body_models fold should look like this
```
./body_models/
├── dmpls/
├──── female/
├──── male/
├──── neutral/
├── smplh/
├──── female/
├──── male/
├──── neutral/
├── smpl_female.pkl
├── smpl_male.pkl
├── smpl_neutral.pkl
```

Additionally, download the `smpl_male.pkl` from [here](https://smpl.is.tue.mpg.de/) and put it in the `./models/` fold.

### AMASS dataset
Please download the AMASS dataset from [here](https://amass.is.tue.mpg.de/). Then run the following command to process AMASS dataset:
```
python prepare_data.py --support_dir /path/to/your/smplh/dmpls --save_dir ./datasets/AMASSIMU/ --root_dir /path/to/your/amass/dataset
```
The generated dataset should look like this
```
./datasets/AMASSIMU/
├── BioMotionLab_NTroje/
├──── train/
├──── test/
├── CMU/
├──── train/
├──── test/
└── MPI_HDM05/
├──── train/
└──── test/
```

### Xsens dataset
We used the same datasets as [DynaIP](https://https://github.com/dx118/dynaip), which include AnDy, UNIPD-BPE, Emokine, CIP and Virginia Natural Motion. These datasets can be downloaded from:

+ AnDy: https://zenodo.org/records/3254403 (xsens_mvnx.zip)
+ UNIPD: https://doi.org/10.17605/OSF.IO/YJ9Q4  (we use all the .mvnx files in single_person folder)
+ Emokine: https://zenodo.org/records/7821844
+ CIP: https://doi.org/10.5281/zenodo.5801928  (MTwAwinda.zip)
+ Virginia Natural Motion: https://doi.org/10.7294/2v3w-sb92
+ DIP-IMU: https://dip.is.tue.mpg.de

Place these datasets in `./datasets/raw/`.

The xsens dataset folder should look like this
```
./datasets/
├── extract
├── raw
├──── andy
├──── cip
├──── dip
├──── emokine
├──── unipd
├──── virginia
├── work
```

Follow the steps below to process raw Xsens datasets.
1. Run `extract.py` , this will extract imu and pose data from raw .mvnx files, downsampling them to 60Hz.  
   

2. Run `process.py` to preprocess IMU data from the extracted Xsens datasets and the raw DIP-IMU dataset. 

### TotalCapture dataset
Download TotalCapture dataset from [here](https://cvssp.org/data/totalcapture/). The TotalCapture dataset fold should look like this

```
./TotalCapture/
├── DIP_recalculate
├── gt_official
├── official
```

Finally, modify the corresponding variables in preprocess.py from lines 19 to 24 according to your own settings and run it to process the AMASS training set and the TotalCapture testing set.

## Training
Run train.py to train the model. Modify the variables from lines 15 to 26 according to different training protocols. The training includes three dataset protocols:
1. Training and testing on the Xsens datasets. Set the `dataset_path_xsens` as your own dataset path and set the `datatype=xsens, totalcapture=False`. The `lr` set as `3e-4`.
2. Training on the AMASS dataset and testing on the TotalCapture dataset. Set the `dataset_path_amass` as the generated dataset path from preprocess.py and set the `datatype=amass, totalcapture=True`. The `lr` set as `1e-4`.
3. Training and testing on the AMASS dataset. Set the `dataset_path_amass` as the generated dataset path from prepare_data.py and set the `datatype=amass, totalcapture=False`. The `lr` set as `3e-4`.

### Stage 1: train VQ-VAE model
Set `pretrained_vqcm=False` and run the following command:
```
python train.py --save_dir ./saved_models/ --batch_size 256 --lr 3e-4(or 1e-4) --save_interval 1000 --log_interval 20 --num_workers 1 --lr_anneal_steps 40000
```

### Stage 2: train 3SIP model
Set `pretrained_vqcm=True`, set `vqcm_path` as your path of the trained VQ-VAE model in stage1, and run the following command:
```
python train.py --save_dir ./saved_models/ --batch_size 256 --lr 3e-4(or 1e-4) --save_interval 1000 --log_interval 20 --num_workers 1 --lr_anneal_steps 40000
```

### Stage 3: adjust pose fusion weight
You can use manual adjustment method or the automatic search method in paper. 
1. To manually adjust pose fusion weight, modify the `c` paramter in the `online()`of `ThreeSIP_divide` in `./model/models.py`.
2. To automatically adjust pose fusion weight, set `threesip_path` as your path of the trained 3SIP model in stage 2, and run the following command:
```
python train.py --save_dir ./saved_models/ --batch_size 256 --lr 1e-4 --save_interval 1000 --log_interval 20 --num_workers 1 --lr_anneal_steps 40000
```
## Evaluation
Run eval.py to evaluate the model. Modify the variables from lines 20 to 33 according to different evaluation protocols. The evaluation includes three dataset protocols:
1. Testing on the Xsens datasets. Set `dataset_path_xsens` as your own dataset path, set `model_path_xsens` as the saved model path in training stage 2, and set `datatype=xsens, totalcapture=False`.
2. Testing on the TotalCapture dataset. Set `dataset_path_amass` as the generated dataset path from preprocess.py, set `model_path_amass` as the saved model path  in training stage 2, and set `datatype=amass, totalcapture=True`.
3. Testing on the AMASS dataset. Set the `dataset_path_amass` as the generated dataset path from prepare_data.py, set `model_path_amass` as the saved model path in training stage 2, and set the `datatype=amass, totalcapture=False`. 

To evaluate 3SIP with manually adjusted pose fusion weight, set `pretrained_3sip=False`; to evaluate 3SIP with automatically searched pose fusion weight, set `pretrained_3sip=True` and set `w_path_amass` (testing on AMASS or TotalCapture datasets) or `w_path_xsens` (testing on Xsens datasets) as the saved model path in training stage 3.

## License
![CC BY-NC 4.0][cc-by-nc-shield]

The majority of 3SIP code is licensed under CC-BY-NC 4.0, however portions of the project are available under separate license terms:
- [AvatarPoser](https://github.com/eth-siplab/AvatarPoser), [AGRoL](https://github.com/facebookresearch/AGRoL), and [DynaIP](https://github.com/dx118/dynaip) are licensed under the MIT license;
- Human Body Prior is licensed under a custom license for non-commercial scientific research purposes, available at [link](https://github.com/nghorbani/human_body_prior/blob/master/LICENSE).

We thank the authors of these projects for their efforts.

## Citation

    @article{guo_three-sensor_2026,
	title = {Three-{Sensor} {Inertial} {Poser} ({3SIP}): {Full}-body motion generation using only three inertial measurement units},
	doi = {10.1016/j.cag.2026.104585},
	journal = {Computers & Graphics},
	author = {Guo, Zihao and Zhao, Jingbo},
	year = {2026},
	pages = {104585}



