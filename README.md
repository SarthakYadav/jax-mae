# mae-jax
- [Sponsors](#sponsors)
- [About](#about)
- [Installation](#installation)
- [Training](#training)
  - [Pre-training](#pre-training)
  - [Linear probe](#linear-probe)
  - [Fine-tuning](#fine-tuning)
- [What's coming up](#whats-coming-up)
- [Data pipeline](#data-pipeline)
- [On contributing](#on-contributing)
- [References](#references)
- [Acknowledgements](#acknowledgements)

## Sponsors
This work would not be possible without cloud resources provided by Google's [TPU Research Cloud (TRC) program](https://sites.research.google/trc/about/). I also thank the TRC support team for quickly resolving whatever issues I had: you're awesome!

Want to become a sponsor? Feel free to reach out!

## About
An implementation of Masked Autoencoders in [Jax](https://jax.readthedocs.io/en/latest).

**PS:** I'm quite new to using Jax and it's functional-at-heart design, so I admit the code can be a bit untidy at places. 
Expect changes, restructuring, and like the official Jax repository itself says, sharp edges!

## Installation
Just make sure requirements are installed. Will add a pip package soon

```shell
pip install -r requirements.txt
```

## Training
- Currently finalizing training of the ViT-Base configuration. Pre-trained models will be released soon.

### Pre-training

```shell
python main.py --config configs/mae_vit_base.py --workdir $PRETRAIN_OUTPUT_DIR
```

### Linear probe
```shell
python main.py --config configs/vit_base.py --workdir $LINEAR_PROBE_OUTPUT_DIR --mode train --pretrained_dir $PRETRAIN_OUTPUT_DIR
```

### Fine-tuning
Not added yet, soon to come

## What's coming up
- Pretrained `MAE` models and linear probe experiments. (VERY SOON!)
- Better documentation and walk-throughs.

## Data pipeline
- All training is done on custom ImageNet TFRecords, but using [tensorflow-datasets](https://www.tensorflow.org/datasets/api_docs/python/tfds) should be trivial.
- I read tfrecords file names from a pandas csv (don't ask, old habits), but you can easily change that to
  reading from a directory.

## On contributing
- At the time of writing, I've been the sole person involved in development of this work, and quite frankly, would love to have help!
- Happy to hear from open source contributors, both newbies and experienced, about their experience and needs
- Always open to hearing about possible ways to clean up/better structure code.

## References

[1] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár & Ross Girshick, "Masked Autoencoders Are Scalable Vision Learners", [online](https://arxiv.org/abs/2111.06377), 2021.  
[2] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby, "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", In Proc. of [ICLR, 2021](https://openreview.net/forum?id=YicbFdNTTy)

## Acknowledgements
* Authors of [1] for the work and their [official pytorch implementation](https://github.com/facebookresearch/mae)
* `timm` by Ross Wightman for being a reliable source of implementations that the community knows will work.
