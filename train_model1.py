#check_units,f2 adapted from Avcu on page https://github.com/keras-team/keras/issues/5400


import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from resnet1 import ResNet
import config1
from sklearn.metrics import classification_report
from build_dataset1 import list_images, list_files
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
from keras import backend as K
# import os


# def list_images(basepath, contains = None):
#     return list_files(basepath, valid=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)


# def list_files(basepath, valid=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):
#     for (rootdir, dirnames, filenames) in os.walk(basepath):
#         for filename in filenames:
#             if contains is not None and filename.find(contains) == -1:
#                 continue
#             ext = filename[filename.rfind("."):].lower()

#             if ext.endswith(valid):
#                 imagepath = os.path.join(rootdir,filename).replace(" ", "\\ ")
#                 yield imagepath

def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred

def f2(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 5*((precision*recall)/(4*precision+recall+K.epsilon()))

ap = argparse.ArgumentParser()
ap.add_argument("-p","--plot", type=str, default="plot.png", help = "path to output loss/accuracy plot")
args = vars(ap.parse_args())

NUM_EPOCHS = 20
# NUM_EPOCHS = 3
INIT_LR = 1e-1
BS = 32

def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    alpha = baseLR * (1-(epoch / float(maxEpochs))) ** power

    return alpha

totaltrain = len(list(list_images(config1.train_path)))
totaldev = len(list(list_images(config1.dev_path)))
totaltest = len(list(list_images(config1.test_path)))

trainAug = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest"
)

valAug = ImageDataGenerator(rescale=1/255.0)

trainGen = trainAug.flow_from_directory(
    config1.train_path,
    class_mode="categorical",
    target_size=(64,64),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS)

devGen = valAug.flow_from_directory(
    config1.dev_path,
    class_mode="categorical",
    target_size=(64,64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

testGen = valAug.flow_from_directory(
    config1.test_path,
    class_mode = "categorical",
    target_size=(64,64),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

model = ResNet.build(64, 64, 3, 2, (3,4,6), (64, 128, 256, 512), reg=0.0005)
opt = SGD(lr=INIT_LR, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy",f2])

callbacks = [LearningRateScheduler(poly_decay)]
starttime = time.time()
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totaltrain//BS,
    validation_data=devGen,
    validation_steps=totaldev//BS,
    epochs=NUM_EPOCHS,
    callbacks=callbacks)
endtime = time.time()
lasting = endtime - starttime
print("--The model cost %dh %dm %ds to train--." %(lasting//3600,int(lasting%3600//60),int(lasting%60)))

print("--Evaluating the network--")
testGen.reset()
predIdxs = model.predict_generator(testGen, steps=(totaltest//BS+1))

predIdxs=np.argmax(predIdxs,axis=1)
print(classification_report(testGen.classes, predIdxs, target_names=testGen.class_indices.keys()))

N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure(0)
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label = "dev_loss")
plt.plot(np.arange(0,N), H.history["acc"], label = "train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label = "dev_acc")
plt.plot(np.arange(0,N), H.history["f2"], label = "f2")

plt.title("Loss and Accuracy on Dataset")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])

# plt.figure(1)
# plt.style.use("ggplot")
# plt.plot(np.arange(0,N), H.history["f2"], label = "f2")
# plt.title("F2 Score on Dataset")
# plt.xlabel("Epoch")
# plt.ylabel("F2 Score")
# plt.legend(loc="lower left")
# plt.savefig("f2.png")






