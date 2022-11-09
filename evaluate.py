import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import json
from easydict import EasyDict as edict
from materials import CheXpertDataSet, CheXpertTrainer, DenseNet121
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Configuration.
cfg_path = './configuration.json'
output_path = './results/'

checkpoint_frt = './results/m-epoch_3_frt.pth.tar'
checkpoint_lat = './results/m-epoch_3_lat.pth.tar'

pathFileTest_frt = './CheXpert-v1.0-small/test_frt.csv'
pathFileTest_lat = './CheXpert-v1.0-small/test_lat.csv'

class_names = ["Card", "Edem", "Cons", "Atel", "PlEf"]

# Load config.
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
datasetTest_frt = CheXpertDataSet(pathFileTest_frt, nnClassCount, cfg.policy, transformSequence)
datasetTest_lat = CheXpertDataSet(pathFileTest_lat, nnClassCount, cfg.policy, transformSequence)

# Create DataLoaders
dataLoaderTest_frt = DataLoader(dataset = datasetTest_frt, num_workers = 2, pin_memory = True)
dataLoaderTest_lat = DataLoader(dataset = datasetTest_lat, num_workers = 2, pin_memory = True)

# Build DenseNet121 model.
model = DenseNet121(nnClassCount, nnIsTrained).cuda()
model = torch.nn.DataParallel(model).cuda()

# Load model and run test.
outGT_frt, outPRED_frt, outPROB_frt, aurocMean_frt, aurocIndividual_frt = CheXpertTrainer.test(
    model, dataLoaderTest_frt, nnClassCount, checkpoint_frt, class_names, 'frt'
)
outGT_lat, outPRED_lat, outPROB_lat, aurocMean_lat, aurocIndividual_lat = CheXpertTrainer.test(
    model, dataLoaderTest_lat, nnClassCount, checkpoint_lat, class_names, 'lat'
)

# Evaluate all test samples.
gts = torch.cat([outGT_frt, outGT_lat], dim=0)
preds = torch.cat([outPRED_frt, outPRED_lat], dim=0)
aurocIndividual = CheXpertTrainer.computeAUROC(gts, preds, nnClassCount)
aurocMean = np.array(aurocIndividual).mean()
print('<<< Model Test Results: AUROC (all) >>>')
print('MEAN', ': {:.4f}'.format(aurocMean))

for i in range (0, len(aurocIndividual)):
    print(class_names[i], ': {:.4f}'.format(aurocIndividual[i]))
print('')
