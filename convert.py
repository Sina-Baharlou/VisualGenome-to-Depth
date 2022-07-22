"""
Convert Visual-Genome to Depth Maps
This code is a part of our paper called "Improving Visual Relation Detection using Depth Maps" (https://github.com/Sina-Baharlou/Depth-VRD).
Created on July 2019, enhanced on July 2022.
"""
import argparse
import os
import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
import models
import json
import logging
import tqdm
from utils import VGDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
tf.disable_eager_execution()

# -- Default parameters --
VERBOSE = True
DEFAULT_META_FILE = "image_data.json"
DEFAULT_IMAGE_PATH = "vg_images"
DEFAULT_MODEL_PATH = "NYU_FCRN.ckpt"
DEFAULT_OUTPUT_PATH = "depth"
DEFAULT_OUTPUT_FMT = "png"
DEFAULT_CV_SIZE = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_OUTPUT_SCALE_FACTOR = 0.5
DEFAULT_N_ITER = 10
DEFAULT_OFFSET = 0
DEFAULT_UPPER_BOUND = -1
DEFAULT_VG_FOLDERS = ['VG_100K', 'VG_100K_2']
DEFAULT_VERBOSE = True


class Converter:
    def __init__(self,
                 md_filename=DEFAULT_META_FILE,
                 img_path=DEFAULT_IMAGE_PATH,
                 model_path=DEFAULT_MODEL_PATH,
                 output_path=DEFAULT_OUTPUT_PATH,
                 output_fmt=DEFAULT_OUTPUT_FMT,
                 cv_size=DEFAULT_CV_SIZE,
                 batch_size=DEFAULT_BATCH_SIZE,
                 output_scale_factor=DEFAULT_OUTPUT_SCALE_FACTOR,
                 n_iterations=DEFAULT_N_ITER,
                 offset=DEFAULT_OFFSET,
                 upper_bound=DEFAULT_UPPER_BOUND,
                 verbose=DEFAULT_VERBOSE
                 ):
        """
        Initializes the Converter class.
        """
        # -- Get dataset and model parameters --
        self.md_filename = md_filename
        self.img_path = img_path
        self.model_path = model_path
        self.output_path = output_path
        self.output_fmt = output_fmt

        # -- Get conversion parameters --
        self.cv_size = cv_size
        self.batch_size = batch_size
        self.output_scale_factor = output_scale_factor
        self.n_iterations = n_iterations
        self.offset = offset
        self.upper_bound = upper_bound

        # -- Class variables --
        self.network = None
        self.idx_iter = None
        self.img_iter = None

        # Set logging level
        logger.setLevel(level=logging.INFO if verbose else logging.WARN)

        # -- Init Visual-Genome dataset wrapper --
        logger.info("Creating the VG dataset...")
        self.generator = VGDataset(md_filename, img_path, self.cv_size)
        dataset = self.generator.get_tf_dataset(batch_size, offset, upper_bound)
        iterator = dataset.make_one_shot_iterator()
        self.idx_iter, self.img_iter = iterator.get_next()

        # -- Construct the network --
        logger.info("Constructing the conversion network...")
        self.network = models.ResNet50UpProj({'data': self.img_iter}, batch_size, 1, False)

        # -- Create output directory --
        logger.info("Creating output directory...")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        # -- Create output sub-directories --
        logger.info("Creating output sub-directories...")
        for sub_dir in DEFAULT_VG_FOLDERS:
            sub_path = os.path.join(self.output_path, sub_dir)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path, exist_ok=True)

    def start(self):
        """
        Starts the converting process.
        """
        try:
            # -- Create a tensorflow session --
            logger.info("Creating a tensorflow session...")
            with tf.Session() as sess:

                # -- Restore the network's weights from ckpt file --
                logger.info("Loading the network's weights...")
                # saver = tf.train.Saver()
                # saver.restore(sess, self.model_path)
                self.network.load(self.model_path, sess)

                # -- Perform the conversion iterations --
                logger.info("Performing the conversion...")
                for _ in tqdm.trange(self.n_iterations, desc="Converting the RGB images to Depth maps..."):
                    # -- Feed forward the images --
                    images, indices = sess.run([self.network.get_output(), self.idx_iter])
                    # images, indices = sess.run([self.img_iter, self.idx_iter])
                    # -- Post process and save the results --
                    for item_idx in range(len(indices)):
                        self.postprocess_save(indices[item_idx], images[item_idx])
                logger.info("Successfully converted %d batches." % self.n_iterations)

        except Exception as err:
            logger.error("Error occurred while converting the images:")
            logger.error(err)
            exit(-1)

    @staticmethod
    def normalize(array):
        """
        Normalize the array between 0 and 1.

        Args:
            array: Numpy input array.
        Returns:
            A normalized array.
        """
        return (array - array.min()) / (array.max() - array.min())

    def postprocess_save(self, index, image):
        """
        Find the region of interest, crop it, normalize it, resize it, and save it.

        Args:
            index: Item index.
            image: Input image.
        """
        # -- Get item details --
        vg_item = self.generator.get_item(index)
        width, height = vg_item['width'], vg_item['height']
        url = vg_item['url']

        # -- Determine the scaling factor --
        output_size = self.cv_size / 2
        max_dim = np.maximum(width, height)
        scale_factor = output_size / max_dim

        # -- Determine the scaled dimensions --
        scaled_w = int(width * scale_factor)
        scaled_h = int(height * scale_factor)

        # -- Crop the region of interest --
        cropped_image = image[:scaled_h, :scaled_w, 0]

        # -- Normalize and convert the image to uint16 --
        nr_image = np.uint16(self.normalize(cropped_image) * (2 ** 16 - 1))

        # -- Create the output image --
        output_img = Image.new("I", nr_image.T.shape)
        output_img.frombytes(nr_image.tobytes(), 'raw', "I;16")

        # -- Resize the image to the desired scale --
        new_width = int(width * self.output_scale_factor)
        new_height = int(height * self.output_scale_factor)
        output_img = output_img.resize((new_width, new_height), Image.ANTIALIAS)

        # -- Save the output --
        folder, filename = url.split('/')[-2:]
        filename = os.path.splitext(filename)[0]  # get rid of the file extension
        output_path = os.path.join(self.output_path, folder, filename)
        output_img.save("{}.{}".format(output_path, self.output_fmt))


def parse_args():
    """
    Parse the commandline arguments.

    Returns:
        the arguments
    """
    parser = argparse.ArgumentParser(
        description="Converts Visual-Genome dataset to depth maps")
    parser.add_argument("config_file", help="Json configuration file")

    args = parser.parse_args()
    return args


def main():
    """
    The goal is to convert Visual-Genome dataset
    to the corresponding depth maps.
    """
    # -- Parse the arguments --
    args = parse_args()

    # -- Open the json file and parse the content --
    with open(args.config_file) as json_file:
        json_conf = json.load(json_file)

    # -- Start the conversion --
    Converter(json_conf).start()


if __name__ == "__main__":
    main()
