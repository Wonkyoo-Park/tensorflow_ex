import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

@tf.function
def train(train_ds, model, loss_object, optimizer, metric_objects):
    for images, labels in train_ds:
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        metric_objects['train_loss'](loss)
        metric_objects['train_acc'](labels, predictions)

@tf.function
def validation(validation_ds, model, loss_object, metric_objects):
    for images, labels in validation_ds:
        predictions = model(images)
        loss = loss_object(labels, predictions)
        metric_objects['validation_loss'](loss)
        metric_objects['validation_acc'](labels, predictions)


def test(test_ds, model, loss_object, metric_objects, path_dict):
    for images, labels in test_ds:
        predictions = model(images)
        loss = loss_object(labels, predictions)

        metric_objects['test_loss'](loss)
        metric_objects['test_acc'](labels, predictions)

    loss, acc = metric_objects['test_loss'].result().numpy(), metric_objects['test_acc'].result()
    with open(path_dict['cp_path'] + '/test_result.txt', 'w') as f:
        template = 'train_loss:{}\ntrain_acc:{}'
        f.write(template.format(loss, acc*100))

def save_metrics_model(epoch, model, losses_accs, path_dict, save_freq):
    if epoch % save_freq == 0:
        model.save(os.path.join(path_dict['model_path'], 'epoch_' + str(epoch)))

    np.savez_compressed(os.path.join(path_dict['cp_path'], 'losses_accs'),
                        train_losses=losses_accs['train_losses'],
                        train_accs=losses_accs['train_accs'],
                        validation_losses=losses_accs['validation_losses'],
                        validation_accs=losses_accs['validation_accs'])