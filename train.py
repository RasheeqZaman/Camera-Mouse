
import matplotlib.pyplot as plt

from yolov3 import Yolo
import config as cfg
from data import data_generator


"""# Data Generator Intialize"""

train_data_generator = data_generator(cfg.dataset[:cfg.num_train], cfg.input_shape, cfg.output_shapes, cfg.num_classes, cfg.image_extension)
val_data_generator = data_generator(cfg.dataset[cfg.num_train:], cfg.input_shape, cfg.output_shapes, cfg.num_classes, cfg.image_extension)


"""# Model Initialize"""

yolo = Yolo(batch_size=cfg.batch_size)
model = yolo.create_model(cfg.input_shape, cfg.num_classes)


"""# Compile"""

model.compile(optimizer=cfg.optimizer, loss=yolo.calc_loss)
print(model.summary())


"""# Fit"""

print('Train on {} samples, val on {} samples, with batch size {}.'.format(cfg.num_train, cfg.num_val, cfg.batch_size))
history = model.fit(x=train_data_generator[0],
                    y=train_data_generator[1],
                    batch_size=cfg.batch_size,  
                    steps_per_epoch=max(1, cfg.num_train//cfg.batch_size),
                    validation_data=val_data_generator,
                    validation_steps=max(1, cfg.num_val//cfg.batch_size),
                    epochs=50,
                    callbacks=[cfg.logging, cfg.checkpoint, cfg.reduce_lr, cfg.early_stopping],
                    verbose=1)
model.save_weights(cfg.weights_path)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()