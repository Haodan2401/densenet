import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import json
from easydict import EasyDict as edict
from materials import CheXpertDataSet, CheXpertTrainer, DenseNet121
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# Configuration.
cfg_path = './configuration.json'
output_path = './results/'

pathFileTrain_frt = './CheXpert-v1.0-small/train_frt.csv'
pathFileTrain_lat = './CheXpert-v1.0-small/train_lat.csv'
pathFileValid_frt = './CheXpert-v1.0-small/valid_frt.csv'
pathFileValid_lat = './CheXpert-v1.0-small/valid_lat.csv'

class_names = ["Card", "Edem", "Cons", "Atel", "PlEf"]

# Load config.
if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(cfg_path) as f:
    cfg = edict(json.load(f))

nnIsTrained = cfg.pre_trained
trBatchSize = cfg.batch_size
trMaxEpoch = cfg.epochs
imgtransResize = cfg.imgtransResize
nnClassCount = cfg.nnClassCount

# Create a dataset
transformSequence = transforms.Compose(
    [
        transforms.Resize((imgtransResize, imgtransResize)),
        transforms.ToTensor(),
    ]
)
datasetTrain_frt = CheXpertDataSet(pathFileTrain_frt, nnClassCount, cfg.policy, transformSequence)
datasetTrain_lat = CheXpertDataSet(pathFileTrain_lat, nnClassCount, cfg.policy, transformSequence)
datasetValid_frt = CheXpertDataSet(pathFileValid_frt, nnClassCount, cfg.policy, transformSequence)
datasetValid_lat = CheXpertDataSet(pathFileValid_lat, nnClassCount, cfg.policy, transformSequence)

# Create DataLoaders
dataLoaderTrain_frt = DataLoader(dataset = datasetTrain_frt, batch_size = trBatchSize,
                                 shuffle = True, num_workers = 2, pin_memory = True) ###
dataLoaderTrain_lat = DataLoader(dataset = datasetTrain_lat, batch_size = trBatchSize,
                                 shuffle = True, num_workers = 2, pin_memory = True) ###
dataLoaderVal_frt = DataLoader(dataset = datasetValid_frt, batch_size = trBatchSize,
                               shuffle = False, num_workers = 2, pin_memory = True)
dataLoaderVal_lat = DataLoader(dataset = datasetValid_lat, batch_size = trBatchSize,
                               shuffle = False, num_workers = 2, pin_memory = True)

# Build DenseNet121 model.
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Train frontal model
model_num_frt, model_num_frt_each, train_time_frt = CheXpertTrainer.train(
    model, dataLoaderTrain_frt, dataLoaderVal_frt, class_names,
    nnClassCount, trMaxEpoch, output_path, 'frt', checkpoint = None, cfg = cfg
)

# Train lateral model
model_num_lat, model_num_lat_each, train_time_lat = CheXpertTrainer.train(
    model, dataLoaderTrain_lat, dataLoaderVal_lat, class_names,
    nnClassCount, trMaxEpoch, output_path, 'lat', checkpoint = None, cfg = cfg
)

# Print training information.
print('<<< Model Trained >>>')
print('For frontal model,', 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt), 'is the best overall.')
for i in range(5):
    print('For frontal {0},'.format(class_names[i]), 'm-epoch_{0}_frt.pth.tar'.format(model_num_frt_each[i]), 'is the best.')
print('')
print('For lateral model,', 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat), 'is the best overall.')
for i in range(5):
    print('For lateral {0},'.format(class_names[i]), 'm-epoch_{0}_lat.pth.tar'.format(model_num_lat_each[i]), 'is the best.')
print('')
