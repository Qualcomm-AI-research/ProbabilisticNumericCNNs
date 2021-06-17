# Probabilistic Numeric CNNs (PNCNNs)
This repository contains the implementation and experiments for the paper presented at ICLR2021.

**Marc Finzi<sup>1</sup>, Roberto Bondesan<sup>2</sup>, Max Welling<sup>2</sup>, "Probabilistic Numeric Convolutional Neural Networks", ICLR 2021.** [[arxiv]](https://arxiv.org/abs/2010.10876)

<sup>1</sup> New York University (Work done during internship at Qualcomm AI Research)
<sup>2</sup> Qualcomm AI Research (Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.)

<p align="center">
  <img src="https://user-images.githubusercontent.com/12687085/92527059-6a8a8a80-f1f4-11ea-812c-b6aba4b6e27e.png" width=800>
</p>

## Training the Models
Assuming the docker container has been built successfully, you can train the relevant models with the following commands.
The author was able to execute his code in a Docker container based on the image nvidia/cuda:11.0.3-base-ubuntu20.04.
After installing using setup.py, you can train the relevant models with the following commands.


### SuperPixelMNIST-75
<p align="center">
  <img src="https://user-images.githubusercontent.com/12687085/92527557-34013f80-f1f5-11ea-9b11-3c0adca0ff00.png" width=800>
</p>

Train the PNCNN on **SuperPixelMNIST-75**, a variant of MNIST digit classification where images are replaced 
by a collection of 75 irregularly spaced superpixels. The dataset will be automatically downloaded when running the script
for the first time to the directory specified by ```--data_dir``` which defaults to scripts/datasets.
```bash
python scripts/train_superpixel.py
```
The script contains several command line arguments such as ```--data_dir --bs --lr --num_epochs``` that can be specified. 
To see all command line arguments and their default values, run ```python scripts/train_superpixel.py --help```.
As shown in the table below, PNCNN greatly improves over the previous state of the art performance.

|         | PNCNN |  GAT |  GCCP  | SplineCNN  | Monet |
|---------|---------|------|------|------|-------|
|Error|   **1.24**  | 3.81 | 4.2 | 4.78 |  8.89 |


### Reference
If you find our work useful, please cite

```
@inproceedings{finzi2021,  
  title={Probabilistic Numeric Convolutional Neural Networks},  
  author={Marc Anton Finzi, Roberto Bondesan, Max Welling},  
  booktitle={International Conference on Learning Representations},  
  year={2021},  
  url={https://openreview.net/forum?id=T1XmO8ScKim}  
}
```
