# From Adversarial to Unlearnable:  A Paradigm for Invariant Model Fingerprinting Across Model Modifications (MFUE)


## About the Project
**MFUE**  is a model fingerprinting approach that uses unlearnable examples to robustly verify model ownership. Unlike adversarial fingerprints, which are sensitive to model modifications and have limited scope, unlearnable examples exhibit greater stability, maintaining low loss values even after significant model changes. **MFUE**  creates these unlearnable examples by employing adversarial training, simulating model modification randomness, and optimizing the examples across different stages. Additionally, a Monte Carlo-based loss estimation method is introduced to extend **MFUE** 's effectiveness in label-only scenarios, where direct loss values are unavailable.
<br>

## Getting Started
### File Structure 
```
ADV-TRA-master
├── models
│   ├── __init__.py
│   ├── densenet.py
│   ├── layers.py
│   ├── ncsnv2.py
│   ├── normalization.py
│   ├── resnet.py
│   ├── swin.py
│   ├── vgg.py
│   └── vit.py
├── utils
│   ├── __init__.py
│   ├── data.py
│   ├── generate.py
│   ├── generic.py
│   └── verification.py
└── main.py
```
There are several parts of the code:
- `models (fold)`: This folder contains the source code for several models.
- `data.py`: This file mainly contains the preprocessing of the raw dataset, and allocation of the dataset.
- `generate.py`: This file involves UF sample generation. 
- `generic.py`: This file contains the training strategy for the source model. 
- `verification.py`: This file contains the Monte Carlo-based loss estimation method.
- `main.py`: The main function of **MFUE**. 
<br>

### Requirements
* python 3.8.15 
* [pytorch](https://pytorch.org/get-started/locally/) 1.12.0 & torchvision 0.13.0 
* CUDA 11.0 and above are recommended (this is for GPU users)
* numpy 1.23.4
* scikit-learn 0.20.0

Before running the project, make sure you have set up the right environment and installed the above required packages.
<br>

### Hyper-parameters 
The settings of **MFUE** are determined in the parameter **args** in **main.py**. Here, we mainly introduce the important hyper-parameters.
- device: which device to use (CPU or GPU).
- dataset: the dataset used for training the target model. 
- num_train: the number of training data for the source model.
- data_path: The path to the raw data.
- model_path: Where to resume the sorce model.
- adv_radius: The UF perturbation radius.
- adv_train_num: The size of the dataset used for adversarial training.
- num_mimic: The number of mimic models.
- num_UF: The number of UF samples.
- UF_radius: The UF perturbation radius.
- num_pb: The number of probing samples.
- spc_path: The path of the suspect model.
- tolerance_factor: Tolerance factor for matching rate calculation.
<br>

### Run
You could run `main.py` in your python IDE directly.
The example codes below show the workflow to perform a complete fingerprinting process, which is in `main.py`.

```python
def main(args):
    # allocate data
    if not os.path.exists(args.data_path + '/' + args.dataset + '/allocated_data/data_log.pth'):
        utils.allocate_data(args)

    # train the source model
    model = utils.get_arch(args.arch, args.dataset)
    utils.train_model(model, args)

    # generate UF samples
    utils.train_mimic(args)
    utils.generate_UF_samples(args)

    # Monte Carlo Estimation
    matching_rate = utils.MC_estimate(args)
```
<br>

## Note
- The Tiny-ImageNet dataset can be downloaded from http://cs231n.stanford.edu/tiny-imagenet-200.zip.
- A GPU is not required, but we recommend using one to increase the running speed. 
