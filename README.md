# Event-Enhanced Synthetic Aperture Imaging
Code of the **SCIENCE CHINA Information Sciences** 2025 paper "*Event-Enhanced Synthetic Aperture Imaging*".



## Requirements

1. Python 3.8 with the following packages installed:
   * opencv-python==4.6.0.66
   * torch==1.9.0
   * pillow==10.4.0
   * prefetch_generator==1.0.1
2. CUDA
   
   - **CUDA** enabled **GPUs** are required for training. We train and test our code with CUDA 11.1 V11.1.105 on A100 GPUs.
   
     

## Dataset

1. Our $\text{THU}^\text{ERGB-SAI}$ dataset could be downloaded from https://github.com/lisiqi19971013/event-based-datasets. 

2. Download the pre-trained model from https://cloud.tsinghua.edu.cn/f/9be2a009d5394feba4d9/?dl=1.

   

## Evaluation

1. Run the following code to generate SAI results.

   ```shell
   >>> python test.py --folder "dataset folder" --ckpt "checkpoint path" --opFolder "opFolder"
   ```

   Then, the outputs will be saved in "opFolder".
   
2. Calculate metrics using the following code.

   ```shell
   >>> python calMetric.py --opFolder "opFolder" --dataFolder "dataset folder"
   ```

   The quantitative results will be save in "opFolder/res.txt"

   
## Training
Run **train_frame.py**, **train_ef.py**, and **train_total.py** step by step to train the model. Make sure to change the dataset path and the model weight path for the previous stage in the file.


## Citation

```bib
@article{ergbsai,
    title={Event-Enhanced Synthetic Aperture Imaging}, 
    author={Li, Siqi and Du, Shaoyi and Yong, Jun-Hai and Gao, Yue},
    journal={SCIENCE CHINA Information Sciences}, 
    volume={68},
    number={3},
    pages={134101},
    year={2025},
}
```
