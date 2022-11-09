# 0. Notes
* Necessary python libraries:
```
easydict
numpy
pandas
pillow
scikit-learn
torch
torchvision
```

* Code is inherited from https://github.com/Stomper10/CheXpert.


# 1. Data folder
* Current train/valid folders are just place-holders.
* Please place small dataset into `CheXpert-v1.0-small`, and the folder structure should be
```
CheXpert-v1.0-small/
├── train
├── train.csv
├── valid
└── valid.csv
```

# 2. Data preprocessing
* The goal is to create separated train/val/test for frontal or lateral images.
* Here test data is a copy of val data.

```bash
python data_preprocess.py
```

The split data information is
```
Train data length(frontal): 191027
Train data length(lateral): 32387
Train data length(total): 223414
Valid data length(frontal): 202
Valid data length(lateral): 32
Valid data length(total): 234
Test data length(frontal): 202
Test data length(lateral): 32
Test data length(total): 234
```

And the final preprocessed data folder structure is
```
CheXpert-v1.0-small/
├── README.md
├── test_frt.csv
├── test_lat.csv
├── train
├── train.csv
├── train_frt.csv
├── train_lat.csv
├── valid
├── valid.csv
├── valid_frt.csv
└── valid_lat.csv
```

# 3. Model training
* CMD is
```
python train.py |& tee ./results/train.log
```

* Training log will be saved in `./results/train.log`.


# 4. Model test
* CMD is
```bash
python evaluate.py
```

* The test results are
```
<<< Model Test Results: AUROC (all) >>>
MEAN : 0.8629
Card : 0.7752
Edem : 0.8852
Cons : 0.9147
Atel : 0.8231
PlEf : 0.9163
```
