import os

# dataset_address = "/Users/KentPeng/Documents/project_medical/malaria/cell_images"
dataset_address = "/home/ubuntu/project/project_medical/cell_images"


base_path = 'malaria'

train_path = os.path.sep.join([base_path,'train'])
dev_path = os.path.sep.join([base_path,'dev'])
test_path = os.path.sep.join([base_path,'test'])

train_split = 0.8
dev_split = 0.1