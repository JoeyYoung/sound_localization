"""
    Packet label_wav.py to be used in detector
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

class KwsNNet:
    def __init__(self, wav_path, pb_path, labels_path, input_node='wav_data:0', output_node='labels_softmax:0', labels_show=3):
        # corresponding to FLAGs in label_wav.py
        self.wav = wav_path
        self.graph = pb_path
        self.labels = labels_path
        self.input_name = input_node
        self.output_name = output_node
        self.how_many_labels = labels_show

        """Loads the model and labels, and runs the inference to print predictions."""

        if not self.labels or not tf.gfile.Exists(self.labels):
            tf.logging.fatal('Labels file does not exist %s', self.labels)

        if not self.graph or not tf.gfile.Exists(self.graph):
            tf.logging.fatal('Graph file does not exist %s', self.graph)

        self.labels_list = self.load_labels(self.labels)

        # load graph, which is stored in the default session
        self.load_graph(self.graph)

    def load_graph(self, filename):
        """Unpersists graph from file as default graph."""
        with tf.gfile.FastGFile(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    def load_labels(self, filename):
        """Read in labels, one label per line."""
        return [line.rstrip() for line in tf.gfile.GFile(filename)]

    def do_inference(self):
        # print("call NN model to do inference ...")
        
        if not self.wav or not tf.gfile.Exists(self.wav):
            tf.logging.fatal('Audio file does not exist %s', self.wav)

        with open(self.wav, 'rb') as wav_file:
            wav_data = wav_file.read()
        
        """Runs the audio data through the graph and prints predictions."""
        with tf.Session() as sess:
            # Feed the audio data as input to the graph.
            #   predictions  will contain a two-dimensional array, where one
            #   dimension represents the input image count, and the other has
            #   predictions per class
            softmax_tensor = sess.graph.get_tensor_by_name(self.output_name)
            predictions, = sess.run(softmax_tensor, {self.input_name: wav_data})

            # Sort to show labels in order of confidence
            top_k = predictions.argsort()[-self.how_many_labels:][::-1]

            rank = 0
            for node_id in top_k:
                human_string = self.labels_list[node_id]
                score = predictions[node_id]
                # print('%s (score = %.5f)' % (human_string, score))
                if rank == 0 and score > 0.8 and human_string == "stop":
                    print("wakeup")
                # TODO, set print to only reponde to wake up word
                rank += 1

            return 0

if __name__ == "__main__":
    kwsnn = KwsNNet("records/go.wav", "Pretrained_models/DNN/DNN_M.pb", "Pretrained_models/labels.txt")
    kwsnn.do_inference()

    # "records/go.wav"