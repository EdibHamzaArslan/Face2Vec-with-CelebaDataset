
from .utils import CelebaDownload
from .utils import DataCleaning
from .utils import DataSeperator
from .utils import DataGenerator
import numpy as np

class DataProcess():
    def __init__(self, img_path, data_path, train_path, val_path, test_path, attr_path):
        '''
        # img_path = 'img_align_celeba/img_align_celeba'
        # data_path = 'CelebFaces/'
        # train_path = 'CelebFaces/Train'
        # val_path = 'CelebFaces/Validation'
        # test_path ='CelebFaces/Test'
        # attribute_data_path = 'list_attr_celeba.csv'
        '''
        self.seperator = DataSeperator(img_path, 
                                        data_path, 
                                        train_path, 
                                        val_path, 
                                        test_path, 
                                        attr_path)
        self.train_df = ''
        self.val_df = ''
        self.test_df = ''
        self.BATCH_SIZE = 64

    def download_data(self):
        # it's gona download celeba dataset and unzip it 
        download = CelebaDownload()
        download.start()

    def seperate_data(self,):        
        # Data path is created
        self.seperator.create_data_paths()
        # reading 
        df = self.seperator.read_attribute_data()
        cleaning = DataCleaning(df)
        df = cleaning.replace()

        img_id_list = list(df['image_id'].to_numpy())
        self.seperator.seperate_big_data_to_data_path(img_id_list)

        self.train_df = df[self.seperator.train_start : self.seperator.train_end ]
        self.val_df = df[self.seperator.val_start : self.seperator.val_end]
        self.test_df = df[self.seperator.test_start: self.seperator.test_end]
        cleaning.delete_files()
        

    def create_train_gen(self,):
        ## Data Generator for training
        train_img_file_list = list(self.train_df['image_id'].to_numpy())
        train_list_IDs = np.array(self.train_df.drop('image_id', axis=1))
        train_gen = DataGenerator(self.seperator.train_path, 
                                list_IDs=train_list_IDs, 
                                labels=train_img_file_list, 
                                batch_size=self.BATCH_SIZE, 
                                shuffle=True)
        return train_gen

    def create_val_gen(self,):
        ## Data Generator for validation
        valid_img_file_list = list(self.val_df['image_id'].to_numpy())
        valid_list_IDs = np.array(self.val_df.drop('image_id', axis=1))
        val_gen = DataGenerator(self.seperator.val_path, 
                                list_IDs=valid_list_IDs, 
                                labels=valid_img_file_list, 
                                batch_size=self.BATCH_SIZE, 
                                shuffle=True)
        return val_gen

    def create_test_gen(self,):
        ## Data Generator for testing
        test_img_file_list = list(self.test_df['image_id'].to_numpy())
        test_list_IDs = np.array(self.test_df.drop('image_id', axis=1))
        test_gen = DataGenerator(self.seperator.test_path, 
                                list_IDs=test_list_IDs, 
                                labels=test_img_file_list, 
                                batch_size=self.BATCH_SIZE, 
                                shuffle=True)
        return test_gen
