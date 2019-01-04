# Malaria_ResNet
Malaria dataset from NIH https://ceb.nlm.nih.gov/proj/malaria/cell_images.zip and the codes at https://ceb.nlm.nih.gov/proj/malaria/malaria_cell_classification_code.zip.
config1.py provides the address of the dataset and the train/dev/test split ratio.

build_dataset1.py will split the dataset into three parts as ratio defined in config1.py

resnet1.py consturct the ResNet.

plot_model.py used to plot the ResNet model

train_model1.py used to train the model as well as plot a graph of several parameters over epochs which will be save as "plot.png" as default.

The model took about 50 minutes to train on AWS EC2 g2.2xlarge instance(NVIDIA GRID K520 GPU) for 20 epochs.
Every epoch took about 138s to train and every step of each epoch cost about 223ms.

Got Accuracy = 0.97, Presion = 0.98, Recall = 0.95, F1 = 0.96, F2 = 0.96 on test set(5512 images in total).

![Aaron Swartz](https://github.com/pzqkent/Malaria_ResNet/raw/master/screenshot/Screen%20Shot%202019-01-02%20at%2011.47.24%20PM.png?raw=true)


