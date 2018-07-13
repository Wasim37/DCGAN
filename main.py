import numpy as np
import tensorflow as tf
import os


flags = tf.app.flags
flags.DEFINE_integer("epoch", 30, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0001, "learning_rate")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data", "Root directory of dataset [data]")
flags.DEFINE_string("dataset", "mnist", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for crop, False for crop [False]")

flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")


FLAGS = flags.FLAGS

def main(_):
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)        
    if not os.path.exists(os.path.join(FLAGS.data_dir, FLAGS.dataset)):
        raise Exception('数据集不存在。。请先下载数据集')
    
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    
    with tf.Session(config=run_config) as sess:
        if FLAGS.dataset == 'mnist':
            # 定义模型
            dcgan = DCGAN(
                      sess,
                      input_width=FLAGS.input_width,
                      input_height=FLAGS.input_height,
                      output_width=FLAGS.output_width,
                      output_height=FLAGS.output_height,
                      batch_size=FLAGS.batch_size,
                      sample_num=FLAGS.batch_size,
                      z_dim=FLAGS.generate_test_images,
                      dataset_name=FLAGS.dataset,
                      input_fname_pattern=FLAGS.input_fname_pattern,
                      crop=FLAGS.crop,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir,
                      data_dir=FLAGS.data_dir)            
        
        if FLAGS.train:
            #dcgan.train()
            pass
        else:
            # 测试
            pass
    

if __name__ == "__main__":
    tf.app.run()