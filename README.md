# Malaria_ResNet
config1.py provides the address of the dataset and the train/dev/test split ratio.

build_dataset1.py will split the dataset into three parts as ratio defined in config1.py

resnet1.py consturct the ResNet.

plot_model.py used to plot the ResNet model

train_model1.py used to train the model as well as plot a graph of several parameters over epochs which will be save as "plot.png" as default.

The model took about 50 minutes to train on AWS EC2 g2.2xlarge instance(NVIDIA GRID K520 GPU) for 20 epochs.
Every epcho took about 138s to train.

Got Accuracy = 0.97, Presion = 0.98, Recall = 0.95, F1 = 0.96, F2 = 0.96 on test set(5512 images in total).
