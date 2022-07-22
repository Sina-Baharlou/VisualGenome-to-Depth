"""
Convert Visual-Genome to Depth Maps
This code is a part of our paper called "Improving Visual Relation Detection using Depth Maps" (https://github.com/Sina-Baharlou/Depth-VRD).
Created on July 2019, enhanced on July 2022.
"""
import tensorflow.compat.v1 as tf
import json
import os


class VGDataset:
    def __init__(self, md_filename, img_path, cv_size):
        """
        Initializes the Visual-Genome dataset wrapper.

        Args:
            md_filename: The path to Visual-Genome's `image-data.json` file.
            img_path: The path to Visual-Genome's images folder.
            cv_size: Canvas Size
        """
        # -- Open the meta data file and convert to json type --
        with open(md_filename, 'r') as json_file:
            self.__json_data = json.load(json_file)

        # -- Assign class variables --
        self.__img_path = img_path
        self.__cv_size = cv_size

    def __mapping_func(self, index, rel_path):
        """
        Maps the filenames to the actual resized image tensors.

        Args:
            index: The index of current image.
            rel_path: The relative path of current image.
        Returns:
            A tuple of index and image tensors.
        """
        # -- Read the current image --
        image_string = tf.read_file(rel_path)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)

        # -- Resize the current image (Ratio-preserving) --
        image_resized = tf.image.resize_images(image_decoded,
                                               (self.__cv_size,
                                                self.__cv_size),
                                               preserve_aspect_ratio=True)

        # -- Fill the canvas by padding the image with zeros --
        image_padded = tf.image.pad_to_bounding_box(image_resized, 0, 0,
                                                    self.__cv_size,
                                                    self.__cv_size)

        # -- specify the final shape --
        image_padded.set_shape([self.__cv_size, self.__cv_size, 3])

        return index, image_padded

    def __generator(self, offset, upper_bound):
        """
        Creates a generator function to retrieve the indices
        and relative paths of the images.

        Args:
            offset: Retrieve items with this offset.
            upper_bound: An upper bound to the list of items.
        Returns:
            A generator function.
        """
        # -- Create the sliced dataset --
        sliced_data = self.__json_data[offset:upper_bound]

        # -- For each item, determine it's relative path and index
        for index, item in enumerate(sliced_data, offset):
            folder, filename = item['url'].split('/')[-2:]
            rel_path = os.path.join(self.__img_path, folder, filename)
            yield index, rel_path

    def get_tf_dataset(self, batch_size=32, offset=0, upper_bound=-1):
        """
        Creates a tensorflow dataset of Visual-Genome's images.

        Args:
            batch_size: The dataset batch size.
            offset: Retrieve items with this offset.
            upper_bound: An upper bound to the list of items.
        Returns:
            A tensorflow dataset.
        """
        # -- determine the upper bound if not specified --
        if upper_bound == -1:
            upper_bound = len(self.__json_data)

        # -- Create the dataset from the given generator --
        dataset = tf.data.Dataset.from_generator(self.__generator,
                                                 (tf.int32, tf.string),
                                                 args=[offset, upper_bound])

        # -- Set the mapping function and batch size --
        dataset = dataset.map(self.__mapping_func)
        dataset = dataset.batch(batch_size)

        return dataset

    def get_item(self, index):
        """
        Returns an item's details.

        Args:
            index: The index of the item.
        Returns:
            A dictionary item providing the image's details.
        """
        return self.__json_data[index]
