# 毕设代码说明

This is the code for work in Yue ZHENG's thesis

毕设题目：图像理解中的图像语义描述方法研究

Here we provide 4 projects:

- frog_attri_coco: a baseline model trained and tested on COCO dataset, using the model in [this paper](https://arxiv.org/abs/1506.01144)
- frog_attri_vg: a baseline model trained and tested on Visual Genome dataset, using a model the same with above
- Frog_relation1: a model detecting *relation* from results from object detecting
- Frog_speak2: a language model generate description from scene graph

data preprocessing

- COCO_PROCESS: codes for processing data in COCO, including tokenlization and making dictionary for caption, augmentation of image
- VG_PROCESS: codes for processing data in VG, including tokenlization and making dictionary for caption, extrating relation and scene graph information, augmentation of image

## Installation

All model in these projects are implemented in [Torch](http://torch.ch/), and depends on the following packages: [torch/torch7](https://github.com/torch/torch7), [torch/nn](https://github.com/torch/nn), [torch/nngraph](https://github.com/torch/nngraph), [torch/image](https://github.com/torch/image), [lua-cjson](https://luarocks.org/modules/luarocks/lua-cjson), [qassemoquab/stnbhwd](https://github.com/qassemoquab/stnbhwd), [jcjohnson/torch-rnn](https://github.com/jcjohnson/torch-rnn)

After installing torch, you can install / update these dependencies by running the following:

```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson
luarocks install https://raw.githubusercontent.com/qassemoquab/stnbhwd/master/stnbhwd-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/torch-rnn-scm-1.rockspec
```

### (Optional) GPU acceleration

If have an NVIDIA GPU and want to accelerate the model with CUDA, you`ll also need to install
[torch/cutorch](https://github.com/torch/cutorch) and [torch/cunn](https://github.com/torch/cunn);
you can install / update these by running:

```bash
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
```

### (Optional) cuDNN

If you want to use NVIDIA's cuDNN library, you'll need to register for the CUDA Developer Program (it's free)
and download the library from [NVIDIA's website](https://developer.nvidia.com/cudnn); you'll also need to install
the [cuDNN bindings for Torch](https://github.com/soumith/cudnn.torch) by running

```bash
luarocks install cudnn
```

## Pretrained model

Pretrained models are provided in data/model for all projects repectively


## Model frog_attri_coco

### Preprocessing

To train a new model, you will follow the following steps:

1. Download the raw images and region caption from [the MS COCO website](http://mscoco.org/)
2. data preprocessing: use code in COCO_PROCESS
   2.1 `coco_extract_features.lua` 
         Using resent to get feature. Here we use [Resnet-101](https://github.com/facebook/fb.resnet.torch), you should download the Resnet-101 model first and put our `coco_extract_features.lua` code in the Resnet-101 model path. With output t7 file
   2.2 `coco_process_select_captions.py`
         extract all caption into a single one `json` file, with output `select_caption_train_val2014.json`
   2.3. `coco_process_dic&caption_idx.py` 
         this code use select_caption_train_val2014.json, with output `dic，concept_vocab, caption` of COCO
   2.4. `make_t7_dataset.lua`
         merge both `hdf5` and `t7` above，with output `t7` file including feature,concept_vocab and caption_idx. The outputs are in two splits, `train` and `val`.
3. put t7 file in the path data/, you can change the file name option in `train_opt.lua` 


For more instructions on training see [INSTALL.md](doc/INSTALL.md) in `doc` folder.

### Training

1. Use the script `train_concept.lua` to train the 	CNN part of model
2. 

## Evaluation

In the paper we propose a metric for automatically evaluating dense captioning results.
Our metric depends on [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/README.html), and
our evaluation code requires both Java and Python 2.7. The following script will download
and unpack the METEOR jarfile:

```bash
sh scripts/setup_eval.sh
```

The evaluation code is **not required** to simply run a trained model on images; you can
[find more details about the evaluation code here](eval/README.md).



