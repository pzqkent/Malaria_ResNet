from resnet1 import ResNet
from keras.utils import plot_model

model = ResNet.build(64, 64, 3, 2, (3,4,6), (64, 128, 256, 512), reg=0.0005)
plot_model(model, to_file = "my_resnet.png", show_shapes = True)