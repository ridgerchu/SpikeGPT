# SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks

<p align="center" float="center">
  <img src="https://github.com/ridgerchu/SpikeGPT/blob/master/static/spikegpt.png"/>
</p>

SpikeGPT is a lightweight generative language model with pure binary, event-driven spiking activation units. The arxiv paper of SpikeGPT can be found [here](https://arxiv.org/abs/2302.13939).

If you are interested in SpikeGPT, feel free to join our Discord using this [link](https://discord.gg/gdUpuTJ6QZ)!

This repo is inspired by the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM).

If you find yourself struggling with environment configuration, consider using the Docker image for SpikeGPT available on [Github](https://github.com/eddiem3/SpikeGPT-container).

## Training on Enwik8

1. Download the `enwik8` dataset by visiting the following link:
   [enwik8 dataset](https://drive.google.com/file/d/1aZQSJctBOYXx76Dld-iioD-v1kR4JHtn/view?usp=sharing).

2. Modify the train set, validate set, and test set paths in the `train.py` script to match the directory where you've extracted the files. For example, if you've extracted the files to a directory named `enwik8_data`, your `train.py` script should be updated as follows:

   ```python
   # Set the paths for the datasets
   datafile_train = "path/to/enwik8_data/train"
   datafile_valid = "path/to/enwik8_data/validate"
   datafile_test = "path/to/enwik8_data/test"


## Pre-training on large corpus

1. **Pre-Training on a Large Corpus**: 
   - To begin, pre-tokenize your corpus data. 
   - For custom data, use the [jsonl2binidx tool](https://github.com/Abel2076/json2binidx_tool) to convert your data. 
   - If you prefer pre-tokenized data, consider using pre-tokenized [The Pile](https://huggingface.co/datasets/RichardErkhov/RWKV-LM_pile_binidx_dataset), which is equipped with a 20B tokenizer and is used in GPT-NeoX and Pythia. 
   - If resources are limited, you may use just one file from the dataset instead of the entire collection.

2. **Configuring the Training Script**: 
   - In `train.py`, uncomment line 82 to enable `MMapIndexedDataset` as the dataset class. 
   - Change `datafile_train` to the filename of your binidx file. 
   - Important: Do not include the `.bin` or `.idx` file extensions.

3. **Starting Multi-GPU Training**:
   - Utilize Hugging Face's Accelerate to begin training on multiple GPUs.

## Fine-Tuning on WikiText-103

1. **Downloading Pre-Tokenized WikiText-103**:
   - You can obtain the pre-tokenized WikiText-103 dataset `binidx` file from this [Hugging Face dataset link](https://huggingface.co/datasets/ridger/Wikitext-series-NeoX-tokenizer/tree/main).

2. **Fine-Tuning the Model**:
   - Use the same approach as in pre-training for fine-tuning your model with this dataset.
   - **Important**: Set a smaller learning rate than during the pre-training stage to avoid catastrophic forgetting. A recommended learning rate is around `3e-6`.
   - For the batch size, it's advisable to adjust according to your specific requirements to find an optimal setting for your case.


## Inference with Prompt

You can choose to run inference with either your own customized model or with our pre-trained model. Our pre-trained model is available [here](https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M). This model trained 5B tokens on OpenWebText. 
1. download our pre-trained model, and put it in the root directory of this repo.
2. Modify the  'context' variable in `run.py` to your custom prompt
3. Run `run.py`

## Fine-Tune with NLU tasks
1. run the file in 'NLU' folders
2. change the path in line 17 to the model path


## Citation


If you find SpikeGPT useful in your work, please cite the following source:


```
@article{zhu2023spikegpt,
        title = {SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks},
        author = {Zhu, Rui-Jie and Zhao, Qihang and Li, Guoqi and Eshraghian, Jason K.},
        journal = {arXiv preprint arXiv:2302.13939},
        year    = {2023}
}
```
