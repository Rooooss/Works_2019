#! /usr/bin/python
# -*- coding: utf8 -*-

from settings import PROJECT_ROOT
import os, time, pickle, random, time, sys
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

import tensorflow as tf
import tensorlayer as tl
import PIL
from IPython.display import clear_output
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config
from metrics import PSNR, SSIM, MSSSIM

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(batch_size))


def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    train_lr_img_list2 = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path2, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list2 = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path2, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = tl.vis.read_images(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)
    train_lr_imgs2 = tl.vis.read_images(train_lr_img_list2, path=config.TRAIN.lr_img_path2, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_image = tf.placeholder('float32', [None, 32, 32, 6], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [None, 32, 32, 3], name='t_target_image')
    t_avg_image = tf.placeholder('float32', [None, 32, 32, 3], name='avg_image_input_to_SRGAN_generator')

    net_g = SRGAN_g(t_image, t_avg_image, is_train=True, reuse=False)
    net_d, logits_real = SRGAN_d(t_target_image, is_train=True, reuse=False)
    net_d_, logits_fake = SRGAN_d(net_g.outputs, is_train=True, reuse=True)

    net_g.print_params(False)
    net_g.print_layers()
    net_d.print_params(False)
    net_d.print_layers()

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    t_target_image_224 = tf.image.resize_images(
        t_target_image, size=[224, 224], method=0,
        align_corners=False)  # resize_target_image_for_vgg # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers.html#UpSampling2dLayer
    t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)  # resize_generate_image_for_vgg

    net_vgg, vgg_target_emb = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    _, vgg_predict_emb = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)

    ## test inference
    net_g_test = SRGAN_g(t_image, t_avg_image, is_train=False, reuse=True)
    print('#########################')
    print(t_image.shape[0])

    # ###========================== DEFINE TRAIN OPS ==========================###
    epsilon = tf.placeholder('float32', [None, 1, 1, 1], name='epsilon')
    interpolated_input = epsilon * t_target_image + (1 - epsilon) * net_g.outputs
    net_i, logits_interpolated = SRGAN_d(interpolated_input, is_train=True, reuse=True)
    gradient = tf.gradients(net_i.outputs, [interpolated_input])[0]
    GP_loss = tf.reduce_mean(tf.square(tf.sqrt(tf.reduce_mean(tf.square(gradient), axis=[1, 2, 3])) - 1))

    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    d_loss = d_loss1 + d_loss2 + 10.0 * GP_loss

    g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(vgg_predict_emb.outputs, vgg_target_emb.outputs, is_mean=True)

    g_loss = mse_loss + vgg_loss + g_gan_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    ## Pretrain
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    ## SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    tl.layers.initialize_global_variables(sess)
    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= LOAD VGG ===============================###
    vgg19_npy_path = "vgg19.npy"
    if not os.path.isfile(vgg19_npy_path):
        print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
        exit()
    npz = np.load(vgg19_npy_path, encoding='latin1').item()

    params = []
    for val in sorted(npz.items()):
        W = np.asarray(val[1][0])
        b = np.asarray(val[1][1])
        print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
        params.extend([W, b])
    tl.files.assign_params(sess, params, net_vgg)
    # net_vgg.print_params(False)
    # net_vgg.print_layers()

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:batch_size]
    sample_lr_imgs1 = train_lr_imgs[0:batch_size]
    sample_lr_imgs2 = train_lr_imgs2[0:batch_size]

    crop_imgs = np.concatenate((sample_imgs, sample_lr_imgs1), axis=3)
    crop_imgs = np.concatenate((crop_imgs, sample_lr_imgs2), axis=3)
    print('==============')
    print(crop_imgs.shape)

    crop_imgs = tl.prepro.threading_data(crop_imgs, fn=crop_sub_imgs_fn, is_random=False)
    sample_imgs_384 = crop_imgs[:, :, :, :3]
    sample_imgs_96 = crop_imgs[:, :, :, 3:6]
    sample_imgs_96_2 = crop_imgs[:, :, :, 6:9]
    print(sample_imgs_384.shape)
    print(sample_imgs_96.shape)
    print(sample_imgs_96_2.shape)

    # sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set
    # sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    # sample_imgs_96 = tl.prepro.threading_data(sample_lr_imgs1, fn=downsample_fn)
    # sample_imgs_96_2 = tl.prepro.threading_data(sample_lr_imgs2, fn=downsample_fn)

    sample_imgs_96_t = np.concatenate((sample_imgs_96, sample_imgs_96_2), axis=3)
    # sample_imgs_96_avg = np.add((sample_imgs_96, sample_imgs_96_2), axis=3) / 2.
    sample_imgs_96_avg = (np.add(sample_imgs_96, sample_imgs_96_2)) / 2.
    # print(sample_imgs_96_t.shape)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_96_2, [ni, ni], save_dir_ginit + '/_train_sample_96_2.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit + '/_train_sample_384.png')
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan + '/_train_sample_96.png')
    tl.vis.save_images(sample_imgs_96_2, [ni, ni], save_dir_gan + '/_train_sample_96_2.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan + '/_train_sample_384.png')

    ###========================= initialize G ====================###
    ## fixed learning rate
    # sess.run(tf.assign(lr_v, lr_init))
    # print(" ** fixed learning rate: %f (for init G)" % lr_init)
    # for epoch in range(0, n_epoch_init + 1):
    #     epoch_time = time.time()
    #     total_mse_loss, n_iter = 0, 0
    #
    #     ## If your machine cannot load all images into memory, you should use
    #     ## this one to load batch of images while training.
    #     # random.shuffle(train_hr_img_list)
    #     # for idx in range(0, len(train_hr_img_list), batch_size):
    #     #     step_time = time.time()
    #     #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
    #     #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
    #     #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
    #     #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)
    #
    #     ## If your machine have enough memory, please pre-load the whole train set.
    #
    #     for idx in range(0, len(train_hr_imgs), batch_size):
    #     #for idx in range(0, 20, batch_size): #test code
    #         step_time = time.time()
    #
    #         crop_b = np.concatenate((train_hr_imgs[idx:idx + batch_size], train_lr_imgs[idx:idx + batch_size]), axis=3)
    #         crop_b = np.concatenate((crop_b, train_lr_imgs2[idx:idx + batch_size]), axis=3)
    #         crop_b = tl.prepro.threading_data(crop_b, fn=crop_sub_imgs_fn, is_random=False)
    #         b_imgs_384 = crop_b[:, :, :, :3]
    #         b_imgs_96_1 = crop_b[:, :, :, 3:6]
    #         b_imgs_96_2 = crop_b[:, :, :, 6:9]
    #         # b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
    #         # b_imgs_96_1 = tl.prepro.threading_data(train_lr_imgs[idx:idx + batch_size], fn=downsample_fn)
    #         # b_imgs_96_2 = tl.prepro.threading_data(train_lr_imgs2[idx:idx + batch_size], fn=downsample_fn)
    #         b_imgs_96 = np.concatenate((b_imgs_96_1, b_imgs_96_2), axis=3)
    #
    #         ## update G
    #         errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_96, t_target_image: b_imgs_384})
    #         print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
    #         total_mse_loss += errM
    #         n_iter += 1
    #     print("???")
    #     log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
    #     print(log)
    #     print(sample_imgs_96_t.shape)
    #     ## quick evaluation on train set
    #     if (epoch != 0) and (epoch % 10 == 0):
    #         out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96_t})  #; print('gen sub-image:', out.shape, out.min(), out.max())
    #         print("[*] save images")
    #         tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)
    #
    #     ## save model
    #     if (epoch != 0) and (epoch % 10 == 0):
    #         tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), sess=sess)

    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        # if epoch != 0 and (epoch % decay_every == 0):
        #     new_lr_decay = lr_decay**(epoch // decay_every)
        #     sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
        #     log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
        #     print(log)
        if epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        ## If your machine cannot load all images into memory, you should use
        ## this one to load batch of images while training.
        # random.shuffle(train_hr_img_list)
        # for idx in range(0, len(train_hr_img_list), batch_size):
        #     step_time = time.time()
        #     b_imgs_list = train_hr_img_list[idx : idx + batch_size]
        #     b_imgs = tl.prepro.threading_data(b_imgs_list, fn=get_imgs_fn, path=config.TRAIN.hr_img_path)
        #     b_imgs_384 = tl.prepro.threading_data(b_imgs, fn=crop_sub_imgs_fn, is_random=True)
        #     b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

        ## If your machine have enough memory, please pre-load the whole train set.

        for idx in range(0, len(train_hr_imgs), batch_size):
        # for idx in range(0, len(train_hr_imgs), 5): # test code
            step_time = time.time()

            crop_b = np.concatenate((train_hr_imgs[idx:idx + batch_size], train_lr_imgs[idx:idx + batch_size]), axis=3)
            crop_b = np.concatenate((crop_b, train_lr_imgs2[idx:idx + batch_size]), axis=3)
            crop_b = tl.prepro.threading_data(crop_b, fn=crop_sub_imgs_fn, is_random=False)
            b_imgs_384 = crop_b[:, :, :, :3]
            b_imgs_96_1 = crop_b[:, :, :, 3:6]
            b_imgs_96_2 = crop_b[:, :, :, 6:9]
            # b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            # b_imgs_96_1 = tl.prepro.threading_data(train_lr_imgs[idx:idx + batch_size], fn=downsample_fn)
            # b_imgs_96_2 = tl.prepro.threading_data(train_lr_imgs2[idx:idx + batch_size], fn=downsample_fn)
            b_imgs_96 = np.concatenate((b_imgs_96_1, b_imgs_96_2), axis=3)
            avg_imgs_96 = (np.add(b_imgs_96_1, b_imgs_96_2)) / 2.

            ep = np.random.uniform(low=0.0, high=1.0, size=[b_imgs_384.shape[0], 1, 1, 1])
            ## update D
            errD, _ = sess.run([d_loss, d_optim], {t_image: b_imgs_96, t_avg_image: avg_imgs_96, t_target_image: b_imgs_384, epsilon: ep})
            ## update G
            errG, errM, errV, errA, _ = sess.run([g_loss, mse_loss, vgg_loss, g_gan_loss, g_optim], {t_image: b_imgs_96, t_avg_image: avg_imgs_96, t_target_image: b_imgs_384, epsilon: ep})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, d_loss: %.8f g_loss: %.8f (mse: %.6f vgg: %.6f adv: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG, errM, errV, errA))
            total_d_loss += errD
            total_g_loss += errG
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter,
                                                                                total_g_loss / n_iter)
        print(log)

        print(sample_imgs_96_t.shape)
        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96_t, t_avg_image: sample_imgs_96_avg})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

        ## save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
            if (epoch % 20 == 0):
                tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_%d.npz'.format(tl.global_flag['mode'])% epoch, sess=sess)
                tl.files.save_npz(net_d.all_params, name=checkpoint_dir + '/d_{}_%d.npz'.format(tl.global_flag['mode'])% epoch, sess=sess)


def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list2 = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path2, regx='.*.png', printable=False))
    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    valid_lr_imgs2 = tl.vis.read_images(valid_lr_img_list2, path=config.VALID.lr_img_path2, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)
    # exit()

    ###========================== DEFINE MODEL ============================###
    tf_gen_output = tf.placeholder('float32', [240, 320, 3])
    tf_hr_output = tf.placeholder('float32', [240, 320, 3])
    t_image = tf.placeholder('float32', [1, None, None, 6], name='input_image')
    avg_image = tf.placeholder('float32', [1, None, None, 3], name='average_image')
    net_g = SRGAN_g(t_image, avg_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    for i in range(0, 30):

        imid = i # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡

        valid_lr_img_t1 = tl.prepro.threading_data(valid_lr_imgs, fn=downsample_fn)
        valid_lr_img_t2 = tl.prepro.threading_data(valid_lr_imgs2, fn=downsample_fn)
        valid_lr_img_t1_d = valid_lr_img_t1[imid]
        valid_lr_img_t2_d = valid_lr_img_t2[imid]
        print(valid_lr_img_t1_d.shape)
        print(valid_lr_img_t2_d.shape)
        valid_hr_img = valid_hr_imgs[imid]
        valid_lr_img = np.concatenate((valid_lr_img_t1_d, valid_lr_img_t2_d), axis=2)
        valid_avg_img = (np.add(valid_lr_img_t1_d, valid_lr_img_t2_d)) / 2.

        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
        # valid_lr_img =  tl.prepro.threading_data(valid_lr_img, fn=downsample_fn)  # rescale to ［－1, 1]
        # print(valid_lr_img.min(), valid_lr_img.max())

        size = valid_lr_img.shape
        print(size)
        # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size



        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img], avg_image: [valid_avg_img]})
        print("took: %4.4fs" % (time.time() - start_time))

        print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        tl.vis.save_image(out[0], save_dir + '/{}/valid_gen.png'.format(imid))
        tl.vis.save_image(valid_lr_img_t1_d, save_dir + '/{}/valid_lr_1.png'.format(imid))
        tl.vis.save_image(valid_lr_img_t2_d, save_dir + '/{}/valid_lr_3.png'.format(imid))
        tl.vis.save_image(valid_hr_img, save_dir + '/{}/valid_hr.png'.format(imid))

        # ssim_gen = tf.image.decode_png(save_dir + '/{}/valid_gen.png'.format(imid))
        # ssim_hr = tf.image.decode_png(save_dir + '/{}/valid_hr.png'.format(imid))

        # ssim_gen_output = out[0]
        # ssim_hr_output = valid_hr_img
        # print(ssim_gen_output)
        # print(ssim_hr_output)


        ssim_gen_output = tl.vis.read_image('valid_gen.png', save_dir + '/{}/'.format(imid))
        ssim_hr_output = tl.vis.read_image('valid_hr.png', save_dir + '/{}/'.format(imid))
        print(ssim_gen_output)
        print(ssim_hr_output)

        ssim1 = tf.image.ssim(tf_gen_output, tf_hr_output, max_val=1.0)

        tf_ssim = sess.run(ssim1, feed_dict={tf_gen_output: ssim_gen_output, tf_hr_output: ssim_hr_output})

        # pre_gray = prediction[:, :, 0]
        # frame2_gray = frame2[:, :, 0]
        # ssim = SSIM(pre_gray, frame2_gray).mean()

        print(tf_ssim)

        # out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
        # tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
