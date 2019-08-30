from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-5
config.TRAIN.beta1 = 0.9

## initialize G
#config.TRAIN.n_epoch_init = 100
config.TRAIN.n_epoch_init = 100
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = 'C:/Users/Joohong/Desktop/2019/srgan-fi/data2017/UCF_train_frameT/'
config.TRAIN.lr_img_path = 'C:/Users/Joohong/Desktop/2019/srgan-fi/data2017/UCF_train_frame1/'
config.TRAIN.lr_img_path2 = 'C:/Users/Joohong/Desktop/2019/srgan-fi/data2017/UCF_train_frame3/'

config.VALID = edict()
## test set location
# config.VALID.hr_img_path = 'C:/Users/Joohong/Desktop/2019/srgan-fi/data2017/UCF_valid_frameT/'
# config.VALID.lr_img_path = 'C:/Users/Joohong/Desktop/2019/srgan-fi/data2017/UCF_valid_frame1/'
# config.VALID.lr_img_path2 = 'C:/Users/Joohong/Desktop/2019/srgan-fi/data2017/UCF_valid_frame3/'

config.VALID.hr_img_path = 'C:/Users/Joohong/Desktop/2019/srgan-fi/data2017/UCF_valid_frameT/'
config.VALID.lr_img_path = 'C:/Users/Joohong/Desktop/2019/srgan-fi/data2017/UCF_valid_frame1/'
config.VALID.lr_img_path2 = 'C:/Users/Joohong/Desktop/2019/srgan-fi/data2017/UCF_valid_frame3/'
def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")


