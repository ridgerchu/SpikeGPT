# SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks

<p align="center" float="center">
  <img src="https://github.com/ridgerchu/SpikeGPT/blob/master/static/spikegpt.png"/>
</p>

SpikeGPT is a lightweight generative language model with pure binary, event-driven spiking activation units.

This repo is inspired by the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM).

## Training on Enwik8

1. Download the [enwik8 dataset](https://data.deepai.org/enwik8.zip).
2. Run `train.py`

## Inference with Prompt

1. Modify the  'context' variable in `run.py` to your custom prompt
2. Run `run.py`



## Citation


If you find SpikeGPT useful in your work, please cite the following source:

```
@misc{https://doi.org/10.48550/arxiv.2302.13939,
  doi = {10.48550/ARXIV.2302.13939},
  
  url = {https://arxiv.org/abs/2302.13939},
  
  author = {Zhu, Rui-Jie and Zhao, Qihang and Eshraghian, Jason K.},
  
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
}

```
