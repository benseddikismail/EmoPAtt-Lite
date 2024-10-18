import shutil
import subprocess
import argparse
# import tarfile
import os
import random
import pandas as pd
import numpy as np
# from PIL import Image
import h5py
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

from elements import SEBlock, SpatialTransformerLayer


def main(train_epochs = 30,
         ft_epochs = 100,
         batch_size = 32,
         learning_rate = 1e-3,
         dropout = 0.1):
    
    # random.seed(0)
    # np.random.seed(0)
    # tf.random.set_seed(0)
    # physical_devices = tf.config.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)



    IMG_SHAPE = (224, 224, 3)
    NUM_CLASSES = 7  # Number of emotion classes
    BATCH_SIZE = batch_size #8

    TRAIN_EPOCH = train_epochs # 100
    TRAIN_LR = learning_rate
    TRAIN_ES_PATIENCE = 5
    TRAIN_LR_PATIENCE = 3
    TRAIN_MIN_LR = 1e-6
    TRAIN_DROPOUT = dropout

    FT_EPOCH = ft_epochs
    FT_LR = 1e-5
    FT_LR_DECAY_STEP = 80.0
    FT_LR_DECAY_RATE = 1
    FT_ES_PATIENCE = 20
    FT_DROPOUT = 0.2

    ES_LR_MIN_DELTA = 0.003

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'dataset/FER2013/train',
        labels='inferred',
        label_mode='int',
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        shuffle=True,
        seed=123,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'dataset/FER2013/val',
        labels='inferred',
        label_mode='int',
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        shuffle=False,  
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'dataset/FER2013/test',
        labels='inferred',
        label_mode='int',
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        shuffle=False,
        seed=123,
    )

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomContrast(factor=0.3),
    ])

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
    # val_ds = val_ds.map(lambda x, y: (data_augmentation(x), y))

    # Model Building
    input_layer = tf.keras.Input(shape=IMG_SHAPE, name='universal_input')
    sample_resizing = tf.keras.layers.Resizing(224, 224, name="resize")
    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip(mode='horizontal'),
                                            tf.keras.layers.RandomContrast(factor=0.3)], name="augmentation")
    preprocess_input = tf.keras.applications.mobilenet.preprocess_input

    # stn_layer = SpatialTransformerLayer(output_size=(224, 224))

    backbone = tf.keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    backbone.trainable = False
    base_model = tf.keras.Model(backbone.input, backbone.layers[-29].output, name='base_model')

    self_attention = tf.keras.layers.Attention(use_scale=True, name='attention')
    patch_extraction = tf.keras.Sequential([
        tf.keras.layers.SeparableConv2D(256, kernel_size=4, strides=4, padding='same', activation='relu'),
        tf.keras.layers.SeparableConv2D(256, kernel_size=2, strides=2, padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='valid', activation='relu'),
        SEBlock(256)
    ], name='patch_extraction')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D(name='gap')
    pre_classification = tf.keras.Sequential([tf.keras.layers.Dense(32, activation='relu'),
                                            tf.keras.layers.BatchNormalization()], name='pre_classification')
    prediction_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name='classification_head')

    inputs = input_layer

    x = SpatialTransformerLayer(output_size=(224, 224))(inputs)

    x = sample_resizing(inputs)
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = base_model(x, training=False)

    # x = CBAMLayer()(x) 

    x = patch_extraction(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(TRAIN_DROPOUT)(x)
    x = pre_classification(x)

    x = ExpandDimsLayer(axis=1)(x)  # Expand dimensions to make it 3D
    x = self_attention([x, x])  # Apply Attention
    x = SqueezeLayer(axis=1)(x)  # Squeeze back to 2D


    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs, name='train-head')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=TRAIN_LR, global_clipnorm=3.0), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Callbacks
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=TRAIN_ES_PATIENCE, min_delta=ES_LR_MIN_DELTA, restore_best_weights=True)
    learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=TRAIN_LR_PATIENCE, verbose=0, min_delta=ES_LR_MIN_DELTA, min_lr=TRAIN_MIN_LR)

    # Training
    history = model.fit(train_ds, epochs=TRAIN_EPOCH, validation_data=val_ds, callbacks=[early_stopping_callback, learning_rate_callback])

    # Model Finetuning
    print("\nFinetuning ...")
    unfreeze = 59
    base_model.trainable = True
    fine_tune_from = len(base_model.layers) - unfreeze
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_from:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inputs = input_layer

    x = SpatialTransformerLayer(output_size=(224, 224))(inputs)

    x = sample_resizing(inputs)
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = patch_extraction(x)
    x = tf.keras.layers.SpatialDropout2D(FT_DROPOUT)(x)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(FT_DROPOUT)(x)
    x = pre_classification(x)

    x = ExpandDimsLayer(axis=1)(x)  # Expand dimensions to make it 3D
    x = self_attention([x, x])  # Apply Attention
    x = SqueezeLayer(axis=1)(x)  # Squeeze back to 2D

    scheduler = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=FT_LR, decay_steps=FT_LR_DECAY_STEP, decay_rate=FT_LR_DECAY_RATE)

    x = tf.keras.layers.Dropout(FT_DROPOUT)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs, name='finetune-backbone')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=scheduler, global_clipnorm=3.0),
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Training Procedure
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=ES_LR_MIN_DELTA, patience=FT_ES_PATIENCE, restore_best_weights=True)
    # scheduler_callback = tf.keras.callbacks.LearningRateScheduler(schedule=scheduler)

    history_finetune = model.fit(train_ds, epochs=FT_EPOCH, validation_data=val_ds, callbacks=[early_stopping_callback, tensorboard_callback])

    # # Plotting
    # plt.figure(figsize=(12, 6))

    # # Plot training accuracy
    # plt.plot(history_finetune.history['accuracy'], label='Train Accuracy', color='#110A44')

    # # Plot validation accuracy
    # plt.plot(history_finetune.history['val_accuracy'], label='Validation Accuracy', color='#F1D43E')

    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    # # Plotting
    # plt.figure(figsize=(12, 6))

    # # Plot training loss
    # plt.plot(history_finetune.history['loss'], label='Train Loss', color='#110A44')

    # # Plot validation loss
    # plt.plot(history_finetune.history['val_loss'], label='Validation Loss', color='#F1D43E')

    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()

    # Evaluation
    test_loss, test_acc = model.evaluate(test_ds)
    model.save('model.h5')

    print('Test loss: %.3f accuracy: %.3f' % (test_loss, test_acc))

    # class_names = test_ds.class_names
    # print("Class names:", class_names)

    # predictions = model.predict(test_ds)
    # y_pred = np.argmax(predictions, axis=1)
    # y_true = np.concatenate([y for x, y in test_ds], axis=0)
    # report = classification_report(y_true, y_pred, target_names=class_names)
    # print("Classification Report:")
    # print(report)
    
    # conf_matrix = confusion_matrix(y_true, y_pred)

    # # confusion matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('Confusion Matrix')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epochs', 
                        type=int, 
                        default=30,
                        help= "number of epochs the model needs to be trained for")
    parser.add_argument('--ft_epochs', 
                    type=int, 
                    default=100,
                    help= "number of epochs the feature extractor needs to be fine-tuned for")
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=32,
                        help = "batch size for training")
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default =1e-3,
                        help = "learning rate for training")
    parser.add_argument('--dropout', 
                        type=float, 
                        default=0.1,
                        help = "dropout rate for training")
    
    args = parser.parse_args()
    main(**vars(args))
