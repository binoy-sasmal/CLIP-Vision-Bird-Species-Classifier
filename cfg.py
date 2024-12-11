from yacs.config import CfgNode as CN

def get_cfg_defaults():
    cfg = CN()

    cfg.model = CN()
    cfg.model.name = "clip_vit"
    cfg.model.pre_trained = True
    cfg.model.encoder_type = "ViT-B/32"

    cfg.train = CN()
    cfg.train.batch_size = 32
    cfg.train.learning_rate = 0.0001
    cfg.train.epochs = 10

    cfg.data = CN()
    cfg.data.train_dir = "./data/train"
    cfg.data.val_dir = "./data/val"
    cfg.data.image_size = 224

    cfg.output_dir = "./outputs"

    return cfg
