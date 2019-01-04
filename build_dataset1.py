import config1
import random
import shutil
import os


def list_images(basepath, contains = None):
    return list_files(basepath, valid=(".jpg", ".jpeg", ".png", ".bmp"), contains=contains)


def list_files(basepath, valid=(".jpg", ".jpeg", ".png", ".bmp"), contains=None):
    for (rootdir, dirnames, filenames) in os.walk(basepath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue
            ext = filename[filename.rfind("."):].lower()

            if ext.endswith(valid):
                imagepath = os.path.join(rootdir,filename).replace(" ", "\\ ")
                yield imagepath




imagepaths = list(list_images(config1.dataset_address))
random.seed(10)
random.shuffle(imagepaths)

i = int(len(imagepaths) * config1.train_split)
trainpaths = imagepaths[:i]
testpaths = imagepaths[i:]

i = int(len(trainpaths) * config1.dev_split)
devpaths = trainpaths[:i]
trainpaths = trainpaths[i:]

datasets = [
    ("train", trainpaths, config1.train_path),
    ("dev", devpaths, config1.dev_path),
    ("test", testpaths, config1.test_path)
]

for (group, grouppaths, groupoutput) in datasets:
    print("--building {} set--".format(group))

    if not os.path.exists(groupoutput):
        print("--creating {} folder--".format(groupoutput))
        os.makedirs(groupoutput)

    for grouppath in grouppaths:
        filename = grouppath.split(os.path.sep)[-1]
        label = grouppath.split(os.path.sep)[-2]

        labelpath = os.path.sep.join([groupoutput,label])

        if not os.path.exists(labelpath):
            print("--creating {} directory".format(labelpath))
            os.makedirs(labelpath)

        a = os.path.sep.join([labelpath, filename])
        shutil.copy2(grouppath, a)


