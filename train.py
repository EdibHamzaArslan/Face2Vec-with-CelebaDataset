import tensorflow as tf
import time 
import numpy as np
import os

class TrainingLoop():
    def __init__(self, model, epochs, batch_size=64, checkpoint='Checkpoints', create_checkpoint=True):
        self.model = model 
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
        self.train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        self.train_loss_metric = tf.keras.metrics.Mean()

        self.val_acc_metric = tf.keras.metrics.BinaryAccuracy()
        self.val_loss_metric = tf.keras.metrics.Mean()

        self.epochs = epochs
        self.metrics_names = ['train_acc', 
                            'train_loss', 
                            'validation_acc', 
                            'validation_loss']
        self.batch_size = batch_size
        self.checkpoint = checkpoint

        if create_checkpoint:
          if not os.path.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
    
    @tf.function
    def train_step(self, x, y):
        true_labels = y
        with tf.GradientTape() as tape:
            prediction = self.model(x, training=True)
            loss_value = self.loss_fn(true_labels, prediction)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        
        # Update metrics
        self.train_acc_metric.update_state(true_labels, prediction)
        self.train_loss_metric.update_state(loss_value)

        return loss_value 
    
    @tf.function
    def test_step(self, x, y):
        val_true_labels = y
        val_prediction = self.model(x, training=False)
        loss_value = self.loss_fn(val_true_labels, val_prediction)
        # Updatate metrics
        self.val_acc_metric.update_state(val_true_labels, val_prediction)
        self.val_loss_metric.update_state(loss_value)
        return loss_value
    

    def train(self, train_datagen, val_datagen):
        train_accs = []
        train_losses = []
        val_accs = []
        val_losses = []
        best_losses = [10000]
        val_loss_is_not_improved_counter = 0
        for epoch in range(self.epochs):
            print("\nepoch {}/{}".format(epoch+1, self.epochs))
            start_time = time.time()
            if val_loss_is_not_improved_counter == 10:
              print("Model can't imrove val_loss during last 10 epoch, training is stopped!")
              break
            # Training
            for step, (x_batch_train, y_batch_train) in enumerate(train_datagen):
                train_loss_value = self.train_step(x_batch_train, y_batch_train)

            # Display metrics at the end of each epoch.
            train_acc = self.train_acc_metric.result()
            train_loss = self.train_loss_metric.result()
            train_accs.append(train_acc)
            train_losses.append(train_loss)
            print("-------------------")
            print("Training acc: %.4f" % (float(train_acc),))
            print("Training loss: %.4f" % (float(train_loss),))
            print("-------------------")

            # Reset training metrics at the end of each epoch
            self.train_acc_metric.reset_states()
            self.train_loss_metric.reset_states()

            # Validation
            for x_batch_val, y_batch_val in val_datagen:
                val_loss_value = self.test_step(x_batch_val, y_batch_val)

            val_acc = self.val_acc_metric.result()
            val_loss = self.val_loss_metric.result()
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            print("Validation acc: %.4f" % (float(val_acc),))
            print("Validation loss: %.4f" % (float(val_loss),))
            print("-------------------")

            if val_loss < best_losses[-1]:
              model_name = f'slim_cnn_model.{epoch:03d}.{val_loss:.2f}.h5'
              print("Model is saving " + model_name)
              self.model.save_weights(os.path.join(self.checkpoint, model_name))
              best_losses.append(val_loss)
              val_loss_is_not_improved_counter = 0
            else:
              val_loss_is_not_improved_counter += 1
              print("Model is not saved, didn't pass best val_loss, counter: ", val_loss_is_not_improved_counter)

            self.val_acc_metric.reset_states()
            self.val_loss_metric.reset_states()
            print("Time taken: %.2fs" % (time.time() - start_time))
        return train_accs, train_losses, val_accs, val_losses


