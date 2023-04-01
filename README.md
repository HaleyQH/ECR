## GPU Requirements
We have run all the experiments on a single GPU NVIDIA GeForce GTX 1080 (12 Gb of GPU memory).
 
## Creating the Environment
Before proceeding, we recommend creating a separate environment to run the code, and 
then installing the packages in requirements.txt:

```
conda create -n consistent-el python=3.9
conda activate consistent-el
pip install -r requirements.txt
``` 

Install pytorch that corresponds to your cuda version. The default pytorch installation command is: 

```
pip install torch torchvision torchaudio
```


## Datasets
