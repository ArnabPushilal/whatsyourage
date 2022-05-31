from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import glob




#globals
TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 198
dataset_folder_name = 'UTKFace'
  

dataset_dict = {
    'race_id': {
        0: 'white', 
        1: 'black', 
        2: 'asian', 
        3: 'indian', 
        4: 'others'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}
dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((g, i) for i, g in dataset_dict['race_id'].items())

def parse_dataset(dataset_path, ext='jpg'):
    """
    Used to extract information about our dataset. It does iterate over all images and return a DataFrame with
    the data (age, gender and sex) of all files.
    """
    def parse_info_from_file(path):
        """
        Parse information from a single file
        """
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')

            return int(age), dataset_dict['gender_id'][int(gender)], dataset_dict['race_id'][int(race)]
        except Exception as ex:
            return None, None, None
        
    files = glob.glob(os.path.join(dataset_path, "*.%s" % ext))
    
    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)
        
    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age', 'gender', 'race', 'file']
    df = df.dropna()
    
    return df



class UtkFaceDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, df):
        self.df = df
        
    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
    
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)

        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]
       

        train_up_to = int(train_up_to * TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        
        # converts alias to id
        self.df['gender_id'] = self.df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])
        self.df['race_id'] = self.df['race'].map(lambda race: dataset_dict['race_alias'][race])

        self.max_age = self.df['age'].max()
        
        return train_idx, valid_idx, test_idx

    

class new_data(Dataset):
    
    def __init__(self,df,index):
        
        self.df=df.iloc[index,:]
       

        self.age=df['age']

    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        im = np.array(im) / 255.0
        
        return im
        
    def __getitem__(self, idx):

    
        person = self.df.iloc[idx]
        file = person['file']        
        age = person['age']
        im = self.preprocess_image(file)

        return im.T,age

    def __len__(self):
        
       return len(self.df)

if __name__=="__main__":



    dataset_folder_name = 'UTKFace'
  
    df = parse_dataset(dataset_folder_name)
    
    
    data_generator = UtkFaceDataGenerator(df)
    train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()

    traindata=new_data(df,train_idx)
    valdata=new_data(df,valid_idx)
    testdata=new_data(df,test_idx)
    
    
   

