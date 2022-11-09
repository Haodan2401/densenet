import pandas as pd

# Split train data to frontal/lateral parts.
Traindata = pd.read_csv('./CheXpert-v1.0-small/train.csv')
Traindata_frt = Traindata[Traindata['Path'].str.contains('frontal')].copy()
Traindata_lat = Traindata[Traindata['Path'].str.contains('lateral')].copy()
Traindata_frt.to_csv('./CheXpert-v1.0-small/train_frt.csv', index = False)
Traindata_lat.to_csv('./CheXpert-v1.0-small/train_lat.csv', index = False)
print('Train data length(frontal):', len(Traindata_frt))
print('Train data length(lateral):', len(Traindata_lat))
print('Train data length(total):', len(Traindata_frt) + len(Traindata_lat))

# Split validation data to frontal/lateral parts.
Validdata = pd.read_csv('./CheXpert-v1.0-small/valid.csv')
Validdata_frt = Validdata[Validdata['Path'].str.contains('frontal')].copy()
Validdata_lat = Validdata[Validdata['Path'].str.contains('lateral')].copy()
Validdata_frt.to_csv('./CheXpert-v1.0-small/valid_frt.csv', index = False)
Validdata_lat.to_csv('./CheXpert-v1.0-small/valid_lat.csv', index = False)
print('Valid data length(frontal):', len(Validdata_frt))
print('Valid data length(lateral):', len(Validdata_lat))
print('Valid data length(total):', len(Validdata_frt) + len(Validdata_lat))

# Split test data to frontal/lateral parts.
# Note: test data is the same as val part since we do not have real test data.
Testdata = pd.read_csv('./CheXpert-v1.0-small/valid.csv')
Testdata_frt = Testdata[Testdata['Path'].str.contains('frontal')].copy()
Testdata_lat = Testdata[Testdata['Path'].str.contains('lateral')].copy()
Testdata_frt.to_csv('./CheXpert-v1.0-small/test_frt.csv', index = False)
Testdata_lat.to_csv('./CheXpert-v1.0-small/test_lat.csv', index = False)
print('Test data length(frontal):', len(Testdata_frt))
print('Test data length(lateral):', len(Testdata_lat))
print('Test data length(total):', len(Testdata_frt) + len(Testdata_lat))
