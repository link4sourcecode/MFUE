# From Adversarial to Unlearnable:  A Paradigm for Invariant Model Fingerprinting Across Model Modifications (MFUE)


## About The Project
**ADV-TRA** is an intellectual property (IP) protection framework for DNN models based on mdoel fingerprinting.
It generates several adversarial trajectories as fingerprints to verify the ownership of a suspect model.
<br>

## Getting Started
### File Structure 
```
ADV-TRA-master
├── utils
│   ├── data_process.py
│   ├── utils.py
│   ├── models.py
│   └── adv_gen.py
├── data
└── main.py
```
There are several parts of the code:

- `data_process.py`: This file mainly contains the preprocessing of the raw dataset, and allocation of the dataset.
- `utils.py`: This file contains the training strategy for the source model. 
- `models.py`: This file contains the internal setting including the architecture of the source models.
- `adv_gen.py`: This file contains the funcations of the whole fingerprinting scheme, including generating the advsarial trajectories as well as verifying the  suspect model with the trajectories.
- `data` folder: Please place the dataset under this folder.
- `main.py`: The main function of **ADV-TRA**. 
<br>

### Requirements

* python 3.8.15 
* [pytorch](https://pytorch.org/get-started/locally/) 1.12.0 & torchvision 0.13.0 
* CUDA 11.0 and above are recommended (this is for GPU users)
* numpy 1.23.4
* advertorch 0.2.3
* scikit-learn 0.20.0

Before running the project, make sure you have set up the right environment and installed the above required packages.
<br>

### Hyper-parameters 
The settings of **ADV-TRA** are determined in the parameter **args** in **main.py**. Here, we mainly introduce the important hyper-parameters.
- device: which device to use (CPU or GPU).
- dataset: the dataset used for training the target model. 
- num_train: the number of training data for the source model.
- num_attack: the number of data for lauching removal attacks.
- initial_lr: initial learning rate for the optimizer.
- epochs: epochs of source model training.
- num_trajectories: the number of trajectories, i.e., fingerprints.
- length: the length of trajectories.
- factor_lc: length control factor to adjust the step size of each step.
- factor_re: reduction factor.
- threshold: threshold for fingerprint determination.
- tra_classes: the number of classes traversed by the trajectory.
- suspect_path: the path of the suspect model.
<br>

### Run
You could run `main.py` in your python IDE directly.
The example codes below show the workflow to perform a complete fingerprinting process, which is in `main.py`.

```python
def main(args):
    # data split
    allocate_data(args)
    
    # Train the source model
    model = build_model(args)
    model = train_model(model, args)
    
    # Generate fingerprints
    generate_trajectory(args)
    
    # verify
    verify_trajectory(args)
```
<br>

## Note
- The ImageNet dataset can be downloaded from https://image-net.org/.
- A GPU is not required, but we recommend using one to increase the running speed. 
