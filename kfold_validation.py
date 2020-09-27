# -*- coding: utf-8 -*-
"""
Created on March 06 2019
tfrecord usage as explained on https://www.tensorflow.org/tutorials/load_data/tf_records
@author: jrbru
description: training and testing a neural network
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import yaml
import os


# ending of the stored results' filenames
validation_version = int(os.environ['PBS_JOBNAME'])
# zero based conv_layer_indices
conv_layer_indices = [2, 7, 10, 13, 14, 19, 22, 25, 29, 32, 35, 39, 42, 45, 46, 51, 54, 57, 61, 64, 67, 71, 74, 77, 81, 84, 87, 88, 93, 96, 99, 103, 106, 109, 113, 116, 119, 123, 126, 129, 133, 136, 139, 143, 146, 149, 150, 155, 158, 161, 165, 168, 171]
# version of the tfrecord dataset to be used
dataset_version = 0
# height = width of an image =
IMAGE_SIZE = 500
# number of folds that the dataset is separated in
k = 10
# number of training epochs = num_epochs
num_epochs = 10
# =============================================================================
# # first layer of pretrained model that is frozen
# first_frozen = 0
# # last layer of pretrained model that is frozen
# last_frozen = -4
# =============================================================================
# number of images in the tfrecord file
# tf.shape(tensor)[0] could be used to get len of tf.tensor
# len(PI_annotations17) should be the same
dataset_size = 31872
batch_size = 16
fold_indices = range(k)
PI_arrays = []
PI_labels = []

def _parse_function(example_proto):
    ''' tf.tensors can be parsed using the function below.
    Alternatively, you can use tf.parse example to parse a whole batch at once.
    '''
    # Feature columns describe how to use the input.
    my_feature_columns = []
    my_feature_columns.append(tf.feature_column.numeric_column(key='feature0'))
    # labels are not features
    # Create a dictionary of the features. 
    feature_dict = tf.estimator.classifier_parse_example_spec(
    my_feature_columns, label_key='labels', label_dtype=tf.int64)

    # Parse the input tf.Example proto using the dictionary above.
    # Or use tf.parse example to parse a whole batch of examples at once
    # return tf.parse_example(example_proto, feature_description)
    return tf.parse_single_example(example_proto, feature_dict)


def train_input_fn(filenames,fold_size,i_test_fold,num_epochs,k,batch_size):
    '''When using the tf.estimator.Estimator API, the first two phases
    (Extract and Transform) are captured in the input_fn passed to
    tf.estimator.Estimator.train.'''
    ''' a tf.data.Dataset is created by loading a list of tfrecords from persistent
    storage.'''
    dataset = tf.data.TFRecordDataset(filenames)
 
    def parser(record):
        featdef = {
            'feature0': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'feature1': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'labels': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        }
 
        example = tf.parse_single_example(record, featdef)
        im = tf.image.decode_jpeg(example['feature0'])
        lbl = example['labels']
        return im, lbl
 
    dataset = dataset.map(parser)
    if i_test_fold == 0:
        trainingset = dataset.skip(fold_size*(i_test_fold+1))
    elif i_test_fold == (k-1):
        trainingset = dataset.take(fold_size*(i_test_fold))
    else:
        trainingset = dataset.take(fold_size*(i_test_fold))
        trainingset_2 = dataset.skip(fold_size*(i_test_fold+1))
        trainingset.concatenate(trainingset_2)
    trainingset = trainingset.batch(batch_size)
    '''The tf.data API provides a software pipelining mechanism through the
    tf.data.Dataset.prefetch transformation, which can be used to decouple
    the time data is produced from the time it is consumed. In particular,
    the transformation uses a background thread and an internal buffer to
    prefetch elements from the input dataset ahead of the time they are
    requested. Thus, to achieve the pipelining effect illustrated above,
    you can add prefetch(1) as the final transformation to your dataset
    pipeline (or prefetch(n) if a single training step consumes n elements).'''
    trainingset = trainingset.prefetch(buffer_size=1)
    trainingset = trainingset.repeat(count=None)
    iterator = trainingset.make_one_shot_iterator()
    features, labels = iterator.get_next()
 
    return ({'resnet50_input': features}, labels)


def test_input_fn(filenames,fold_size,i_test_fold,k,batch_size):
    '''When using the tf.estimator.Estimator API, the first two phases
    (Extract and Transform) are captured in the input_fn passed to
    tf.estimator.Estimator.train.'''
    ''' a tf.data.Dataset is created by loading a list of tfrecords from persistent
    storage.'''
    dataset = tf.data.TFRecordDataset(filenames)
 
    def parser(record):
        featdef = {
            'feature0': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'feature1': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'labels': tf.FixedLenFeature(shape=[], dtype=tf.int64),
        }

        example = tf.parse_single_example(record, featdef)
        im = tf.image.decode_jpeg(example['feature0'])
        lbl = example['labels']
        return im, lbl
 
    dataset = dataset.map(parser)
    if i_test_fold == 0:
        testset = dataset.take(fold_size)
    elif i_test_fold == (k-1):
        testset = dataset.skip(fold_size*(i_test_fold))
    else:
        testset = dataset.skip(fold_size*(i_test_fold))
        testset = testset.take(fold_size)
    testset = testset.batch(batch_size)
    '''The tf.data API provides a software pipelining mechanism through the
    tf.data.Dataset.prefetch transformation, which can be used to decouple
    the time data is produced from the time it is consumed. In particular,
    the transformation uses a background thread and an internal buffer to
    prefetch elements from the input dataset ahead of the time they are
    requested. Thus, to achieve the pipelining effect illustrated above,
    you can add prefetch(1) as the final transformation to your dataset
    pipeline (or prefetch(n) if a single training step consumes n elements).'''
    testset = testset.prefetch(buffer_size=1)
    testset = testset.repeat(count=None)
    iterator = testset.make_one_shot_iterator()
    features, labels = iterator.get_next()
 
    return ({'resnet50_input': features}, labels)


# body
filename = 'PI/PI_dataset' + str(dataset_version) + '.tfrecords'
# a list that only contains the filename of the dataset to be loaded is created
filenames = [filename]

'''separate specified amounts of data for training and testing'''
fold_size = round(dataset_size / k)
trainingset_size = dataset_size - fold_size

# start kfold validation
i_test_fold = int(os.environ['PBS_ARRAY_INDEX'])

# load resnet with pretrained weights of imagnet
pretrained_resnet = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
)

# freeze the conv layers
for i_layer in conv_layer_indices[:validation_version]:
    pretrained_resnet.layers[i_layer].trainable = False

model = tf.keras.models.Sequential()
model.add(pretrained_resnet)
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(645, activation='softmax'))

sgd = tf.keras.optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Check the trainable status of the individual layers
print('List of transferred layers with trainable status:')
for layer in pretrained_resnet.layers:
    print(layer, layer.trainable)

print(model.summary())
print('Model metric names:')
print(model.metrics_names)

x_train, y_train = train_input_fn(filenames,fold_size,i_test_fold,num_epochs,k,batch_size)
x_test, y_test = test_input_fn(filenames,fold_size,i_test_fold,k,batch_size)

print('Start training for fold ' + str(i_test_fold))
history = model.fit(x_train, y_train, epochs=num_epochs, steps_per_epoch=round(trainingset_size/batch_size), validation_data=(x_test, y_test), validation_steps=round(fold_size/batch_size), verbose=2)
print('Start testing with fold ' + str(i_test_fold))
trainings=history.history['sparse_categorical_accuracy']
evaluations=history.history['val_sparse_categorical_accuracy']
# alternative to validation model.evaluate(x_test, y_test, steps=round(fold_size/batch_size))
# Save the model
model.save('Results/PI_resnet_V' + str(validation_version) + 'F' + str(i_test_fold) + '.h5')

print('training accuracies:')
print(trainings)
print('validation accuracies:')
print(evaluations)
print('Outer list contains a list for each fold, while inner lists contains an accuracy for each epoch.')

trainings_key='trainings_V'+str(validation_version)+'F'+str(i_test_fold)
evalutations_key='evaluations_V'+str(validation_version)+'F'+str(i_test_fold)
dictionary = {trainings_key: trainings,
              evalutations_key: evaluations,
              }
# saving all lists by serialising to yaml
for key, value in dictionary.items():
    stream = open('Results/' + key + '.yaml', 'w')
    yaml.dump(value, stream)
