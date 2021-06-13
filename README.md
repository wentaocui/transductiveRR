# Parameterless Transductive Re-representation In Few-Shot Learning


##  Introduction
This repository maintains the code of our ICML 2021 paper "Parameterless Transductive Re-representation In Few-Shot Learning". Our framework applies a parameterless transductive feature re-representation (RR) to achieve better feature representation and is compatible with most few-shot learning methods. The experiments in our paper include applying RR to the following baselines. The codes of these baselines are reused. 
* Transductive Infomation Maximization (TIM) for few-shot learning. 
  * During inference, we applied RR to TIM to create TIM+RR.
  * https://github.com/mboudiaf/TIM
* Laplacian Regularized Few-Shot Learning. 
  * During inference, we applied RR to LaplacianShot to create LaplacianShot+RR.
  * https://github.com/imtiazziko/LaplacianShot
* SimpleShot: Revisiting Nearest-Neighbor Classification for Few-Shot Learning. 
  * During inference, we applied RR to SimpleShot to create SimpleShot+RR.
  * https://github.com/mileyan/simple_shot
* A Closer Look at Few-shot Classification. 
  * We applied RR during inference to meta-learning baselines [ProtoNet](https://arxiv.org/abs/1703.05175), [Relation](https://arxiv.org/abs/1711.06025) and [Matching](https://arxiv.org/abs/1606.04080) to create ProtoNet+RR, Relation+RR, Matching+RR
  * We applied RR and self-supervised learning during meta-training (MT) to [ProtoNet](https://arxiv.org/abs/1703.05175), [Relation](https://arxiv.org/abs/1711.06025) and [Matching](https://arxiv.org/abs/1606.04080) to create ProtoNet+MT, Relation+MT, Matching+MT
  * https://github.com/wyharveychen/CloserLookFewShot
  

## 1. Installation

The code repositories of all baselines are merged in this single repository. The folders for each baseline are independent. The backbones and datsets of TIM, LaplacianShot and SimpleShot are identical and can be reused.

We performe the following experiments using Python 3.6 and Pytorch 1.4.0. Please ensure the same Python and Pytorch are used.

### 1.1 Download code

Pulling from this code repository will download code for all baselines, except saved models and datasets.

### 1.2 Download datasets

The code to experiment meta-learning models is under few-shot-master/. To download mini-Imagenet dataset, we reuse the script from the [project page](https://github.com/hytseng0509/CrossDomainFewShot) of paper "Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation".

```
cd few-shot-master/filelists
python3 process.py miniImagenet
cd ..
```

To download mini-Imagenet, tiered-Imagenet and CUB datasets for TIM, SimpleShot and LaplacianShot, refer to the README.md of individual subfolder. The data format and split of the three models are identical, so one can download just once and reuse them. Simply change the data path in the configuration Python files as needed, or copy dataset folders into corresponding destination folders.


### 1.3 Download models

Pre-trained backbones of SimpleShot, TIM and LaplacianShot can be downloaded by running model download scripts as detailed in each README file. The backbones can also be shared among these three models since they are of the same architecture and are pre-trained by the same datasets. Downloading these pre-trained backbones is required before running RR.

CUB trained WRN backbone is not provided by all baseline projects. We pre-train WRN on CUB and share it in model download link (see below). Place the folder "wideres" under proper folders for all three non-meta-learning baselines.

We provide mini-Imagenet pre-trained models for ProtoNet, Relation and Matching, using https://github.com/wyharveychen/CloserLookFewShot, and the trained models for ProtoNet+MT, Relation+MT, Matching+MT. Please use this [download link](https://drive.google.com/drive/folders/1fPk6xg6MXFOfP8PcD4iqa6-_Woidv2On?usp=sharing). Place the downloaded models under RR/few-shot-master/checkpoints/miniImagenet/<your_specific_model_name>/. The absolute path can be found after RR_save_features.py line 66 in variable "checkpoint_dir". 


## 2. Train models

Applying RR to non-meta-learning baselines (TIM, LaplacianShot, SimpleSHot) does not require training, since RR simply works as a feature re-representation layer between the backbone and the classifier. We do not provide a saved baseline model since we also reuse the ones baseline papers provide.

RR can be applied to meta-learning baselines (ProtoNet, Matching, Relation) during inference, similar to applying it to non-meta-learning baselines, and this does not require training given pre-trained baseline models. When RR is applied during meta-training to meta-learning baselines, baseline+MT will be created. In the model download link, we provide both the pre-trained baselines (baseline_model.tar) and baseline+MT trained models (best_model.tar).


### 2.1 Train meta-learning baselines with MT (RR+SSL)

Meta-learning baseline + RR/MT requires an available trained baseline model. We provide such trained models, named as "baseline_model.tar" under each model folder. One could also train the meta-learning baselines from scratch by following the instructions of "few-shot-master" README.md.

To train baseline+MT, execute the following commands. Our paper only experiments 5-way tests on mini-Imagenet dataset with ResNet-18 backbone. Adjustable parameters in the command include the following. Check few-shot-master/io_utils.py for details.
* method: transductProtoNet, transductMatchingNet, transductRelationNet.
* n_shot: 1, 5.

```
cd few-shot-master/
python RR_train.py --dataset miniImagenet --model ResNet18 --method transductProtoNet --train_n_way 5 --test_n_way 5 --n_shot 1 --transduct_mode MT
```

## 3. Evaluate models

### 3.1 Meta-learning baseline + RR/MT

Adjustable parameters in the command include the following. Check few-shot-master/io_utils.py for details. To run all experiments in the paper, check few-shot-master/run.sh.
* method: transductProtoNet, transductMatchingNet, transductRelationNet.
* n_shot: 1, 5.
* transduct_mode: off, RR, MT.

```
cd few-shot-master/
python RR_save_features.py --dataset miniImagenet --model ResNet18 --method transductProtoNet --train_n_way 5 --test_n_way 5 --n_shot 1 --transduct_mode MT
python RR_test.py --dataset miniImagenet --model ResNet18 --method transductProtoNet --train_n_way 5 --test_n_way 5 --n_shot 1 --transduct_mode MT
```

### 3.2 SimpleShot + RR

Adjustable parameters in the command include the following. Check simple_shot/src/utils/configuration.py for details. To run all experiments in the paper, check simple_shot/src/runRR.sh.
* data-for-all: miniImagenet, tiered_imagenet, cub.
* arch-for-all: resnet18, wideres.
* meta-val-way: 5, 10, 20. Note that 10 and 20 way test was only with mini-Imagenet and ResNet-18. 20-way tests reuse 10-way hyperparameters: alpha1, alpha2 and tau, since validation set class number is less than 20.
* eval-shot: 1, 5.
* enable-transduct: true (i.e., RR) if flag is present, otherwise false (i.e., off).

```
cd simple_shot/
python ./src/train.py --data-for-all miniImagenet --arch-for-all resnet18 --meta-val-way 5 --eval-shot 1  --enable-transduct
```

### 3.3 LaplacianShot + RR

Adjustable parameters in the command are similar to SimpleShot+RR. Check laplacianshot/src/utils/configuration.py for details. To run all experiments in the paper, check laplacianshot/runRR.sh.

```
cd laplacianshot/
python ./src/train_lshot.py --data-for-all miniImagenet --arch-for-all resnet18 --meta-val-way 5 --eval-shot 1  --enable-transduct
```

### 3.4 TIM + RR

Adjustable parameters in the command are similar to SimpleShot+RR. Check tim/src/utils.py for details. To run all experiments in the paper, check tim/run.sh.

```
cd tim/
python src/main.py --data-for-all miniImagenet --arch-for-all resnet18 --eval-n-ways 5 --eval-shot 1 --enable-transduct
```

### 3.5 Higher way

We test 10-way and 20-way performances of SimpleShot+RR, LaplacianShot+RR and TIM+RR. To run these tests, simply change "--meta-val-way" or "--eval-n-ways" arguments in the commands in section 3.2-3.4. Check more details in the configuration or utility files in each model folder.

### 3.6 Varying support shots

We test 5-way performances of SimpleShot+RR and TIM+RR with varying support set shots (specifically from 1 through 10). To run these tests, simply change "--eval-shot" arguments in the commands in section 3.2 and 3.4. Check more details in the configuration or utility files in each model folder.

### 3.7 Varying query shots

We test 5-way performances of SimpleShot+RR and TIM+RR with 1 support shot and varying (specifically 5, 10, 15, 20, 25 and 30) query set shots. To run these tests, simply set "--eval-shot" to 1, and set "--meta-val-query" (for SimpleShot) or "--eval-query-shots" (for TIM) as one of [5,10,15,20,25,30] in the commands in section 3.2 and 3.4. Check more details in the configuration or utility files in each model folder.

### 3.8 Iterative RR

We test 5-way 1-shot and 5-shot performances of SimpleShot+RR and TIM+RR with varying rounds of RR, i.e., applying RR different number of times during inference. To run these tests, add "--iterative-transduct" in the command and set "--n-ierative-transduct". We tested "--n-ierative-transduct" from 0 to 4. Examples are given below. Check more details in the configuration or utility files in each model folder.

```
cd simple_shot/
python ./src/train.py --data-for-all miniImagenet --arch-for-all resnet18 --meta-val-way 5 --eval-shot 1 --enable-transduct --iterative-transduct --n-ierative-transduct 1
```

```
cd tim/
python src/main.py --data-for-all miniImagenet --arch-for-all resnet18 --eval-n-ways 5 --eval-shot 1 --enable-transduct --iterative-transduct --n-ierative-transduct 1
```



