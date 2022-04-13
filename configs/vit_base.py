import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.model = ml_collections.ConfigDict()
    config.model.arch = "vit_base_patch16"
    config.model.type = "multiclass"
    config.model.num_classes = 1000
    config.model.pretrained = ""
    config.model.pretrained_fc_only = True

    config.opt = ml_collections.ConfigDict()
    config.opt.optimizer = "adamw"
    config.opt.learning_rate = 0.01
    config.opt.weight_decay = 0.
    config.opt.schedule = "constant"
    config.opt.warmup_epochs = 5.0
    config.opt.momentum = 0.9

    config.log_every_steps = 100
    config.num_train_steps = -1
    config.steps_per_eval = -1

    config.data = ml_collections.ConfigDict()
    config.data.tr_manifest = "/home/sarthak/my_disk/Datasets/imagenet/train.csv"
    config.data.eval_manifest = "/home/sarthak/my_disk/Datasets/imagenet/val.csv"
    config.data.tr_samples = 1281167
    config.data.eval_samples = 50000
    config.data.compression = "ZLIB"
    config.data.reader = "tfio"
    config.data.cacheable = False
    config.data.jax_transforms = True
    config.data.dataset_name = "imagenet2012"

    config.batch_size = 8*128
    config.half_precision = False
    config.input_shape = (224, 224, 3)
    config.num_epochs = 50
    config.device = None

    config.wandb = ml_collections.ConfigDict()
    config.wandb.project = "mae-jax"

    return config
