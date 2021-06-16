
import os
import numpy as np 
import pandas as pd 
import shutil
import tensorflow as tf

class CelebaDownload():
    def __init__(self):
        os.system('mkdir ~/.kaggle')
        os.system('cp data/kaggle.json ~/.kaggle/')
        os.system('chmod 600 ~/.kaggle/kaggle.json')
        print('Kaggle.json is successfully setuped.')
    def start(self):
        os.system('kaggle datasets download -p data/ -d jessicali9530/celeba-dataset')
        os.system('unzip data/celeba-dataset.*zip')
        print("Celeba Downloaded")



class DataCleaning():

    def __init__(self, df):
        self.df = df
    
    def delete_files(self):
        '''
        After data is seperated old data
        will deleted by invoking this function
        '''
        os.system('rm -r data/celeba-dataset.zip')
        os.system('rm -r img_align_celeba')
        os.system('rm list_landmarks_align_celeba.csv')
        os.system('rm list_eval_partition.csv')
        os.system('rm list_bbox_celeba.csv')
        print('Downloaded data is deleted')



    def replace(self):
        '''Replacing -1 to 0 in .csv
        dataset
        '''
        return self.df.replace(-1, 0)

    def drop(self, features=['Attractive', 'Blurry', 'Young']):
        self.df = self.df.drop(features, axis=1)
        self.replace()
       
        self.df = self.df[(self.df['Male'] == 0) & 
                          (self.df['5_o_Clock_Shadow'] ==0) & 
                          (self.df['Bald'] == 0) & 
                          (self.df['Goatee']==0) & 
                          (self.df['Mustache'] == 0) & 
                          (self.df['Sideburns'] == 0) & 
                          (self.df['Wearing_Necktie'] == 0)]
       
        self.df = self.df.drop(['5_o_Clock_Shadow', 
                                'Bald', 
                                'Goatee', 
                                'Male', 
                                'Mustache', 
                                'Sideburns', 
                                'Wearing_Necktie', 
                                'Mouth_Slightly_Open', 
                                'No_Beard', 
                                'Receding_Hairline', 
                                'Smiling'], axis=1)
    
    
    def rearrange_raw_data(self):
        self.new_df = pd.DataFrame()
        self.new_df['image_id'] = self.df['image_id']

        # nose
        self.new_df['Big_Nose'] = self.df['Big_Nose']
        self.new_df['Pointy_Nose'] = self.df['Pointy_Nose']

        # mouth
        self.new_df['Big_Lips'] = self.df['Big_Lips']
        self.new_df['Wearing_Lipstick'] = self.df['Wearing_Lipstick']

        # eyes
        self.new_df['Arched_Eyebrows'] = self.df['Arched_Eyebrows']
        self.new_df['Bags_Under_Eyes'] = self.df['Bags_Under_Eyes']
        self.new_df['Bushy_Eyebrows'] = self.df['Bushy_Eyebrows']
        self.new_df['Eyeglasses'] = self.df['Eyeglasses']
        self.new_df['Narrow_Eyes'] = self.df['Narrow_Eyes']

        # faces
        self.new_df['Heavy_Makeup'] = self.df['Heavy_Makeup']
        self.new_df['Oval_Face'] = self.df['Oval_Face']
        self.new_df['Pale_Skin'] = self.df['Pale_Skin']

        # Others
        # around_head
        self.new_df['Bangs'] = self.df['Bangs']
        self.new_df['Black_Hair'] = self.df['Black_Hair']
        self.new_df['Blond_Hair'] = self.df['Blond_Hair']
        self.new_df['Brown_Hair'] = self.df['Brown_Hair']
        self.new_df['Gray_Hair'] = self.df['Gray_Hair']
        self.new_df['Straight_Hair'] = self.df['Straight_Hair']
        self.new_df['Wavy_Hair'] = self.df['Wavy_Hair']
        self.new_df['Wearing_Earrings'] = self.df['Wearing_Earrings']
        self.new_df['Wearing_Hat'] = self.df['Wearing_Hat']
        self.new_df['Wearing_Necklace'] = self.df['Wearing_Necklace']

        #checks
        self.new_df['High_Cheekbones'] = self.df['High_Cheekbones']
        self.new_df['Rosy_Cheeks'] = self.df['Rosy_Cheeks']

        #fat
        self.new_df['Chubby'] = self.df['Chubby']
        self.new_df['Double_Chin'] = self.df['Double_Chin']

        return self.new_df


class DataSeperator():
    ''' The images are sending out 
    train, val and test directory depend on
    ratio of split.
    '''
    TOTAL_SIZE = 202599
    train_start, train_end = 0, 162599
    val_start, val_end = train_end, train_end + 20000
    test_start, test_end = val_end , val_end + 20000

    def __init__(self, img_path, data_path, train_path, val_path, test_path, attr_path):
        self.img_path = img_path
        self.data_path = data_path
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.attr_path = attr_path


    def read_attribute_data(self):
        return pd.read_csv(self.attr_path)

    def create_data_paths(self):
        
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)

        if not os.path.exists(self.train_path):
            os.mkdir(self.train_path)
        
        if not os.path.exists(self.val_path):
            os.mkdir(self.val_path)

        if not os.path.exists(self.test_path):
            os.mkdir(self.test_path)
        

            
    def copy_files_to_destination_path(self, fnames, dst_dir):
        for fname in fnames:
            src = os.path.join(self.img_path, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copy(src, dst)

            
    def seperate_big_data_to_data_path(self, img_id_list):
        if len(os.listdir(self.train_path)) == 0:
            train_fnames = img_id_list[self.train_start:self.train_end]
            vald_fnames = img_id_list[self.val_start:self.val_end]
            test_fnames = img_id_list[self.test_start:self.test_end]

            self.copy_files_to_destination_path(train_fnames, self.train_path)
            self.copy_files_to_destination_path(vald_fnames, self.val_path)
            self.copy_files_to_destination_path(test_fnames, self.test_path)
            print("Data seperation is successfully.")

        else:
            print('Data is already splitted')





class DataGenerator(tf.keras.utils.Sequence):
    
    ''' Generates data for keras
    Sources:
    https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    '''

    def __init__(self, image_path, list_IDs, labels, batch_size=64, dim=(178, 218), n_channels=3, n_classes=1, shuffle=True):
        self.image_path = image_path
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        if self.batch_size > self.list_IDs.shape[0]:
            print("Batch size is greater than data size!!")
            return -1
        return int(np.floor(self.list_IDs.shape[0] / self.batch_size))

    def __getitem__(self, index):
        
        indexes = self.indexes[index*self.batch_size : (index+1)*self.batch_size]

        image_file_list = [self.labels[k] for k in indexes] # The labels is a list which is contain file names. For instance ["000002.jpg", "000003.jpg", ...]
        list_IDs_temp = np.array([self.list_IDs[k] for k in indexes]).astype(np.float32)
        

        X = self.__generate_X(image_file_list)
        # y = self.__generate_y(list_IDs_temp)
        y = list_IDs_temp

        return X, y

    def __generate_X(self, image_file_list):
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        for i in range(len(image_file_list)):
            img = tf.keras.preprocessing.image.load_img(self.image_path + "/" + image_file_list[i], target_size=self.dim)
            img = tf.keras.preprocessing.image.img_to_array(img)
            # img = np.expand_dims(img, axis=0)
            img = img.astype("float32") / 255
            X[i] = img
        return X


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.seed(2)
            np.random.shuffle(self.indexes)