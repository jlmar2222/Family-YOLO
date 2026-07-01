import torch
from torch.utils.data import Dataset 
import pandas as pd 
import cv2


class VocDataset(Dataset):
    def __init__(self, df : pd.DataFrame, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']     
       
        image = cv2.imread(image_path) # C = 3      
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")        

        
        if self.transforms:
            # Due to Albumentation library, 
            # we need to pass the image as a dictionary 
            # and get the transformed image back from the dictionary
            transformed = self.transforms(image=image)
            image = transformed['image']
        
            

        # get labels
        # if any issue with the column name, 
        # we set the label to 0.0 by default
        sample = {
            'image': image,
            'cancer_label': torch.tensor(row.get('Cancer', 0.0), dtype=torch.float32),
            'age_label': torch.tensor(row.get('Age', 0.0), dtype=torch.float32),
            'density_label': torch.tensor(row.get('Density', 0.0), dtype=torch.float32),             
            'patient_id': row.get('ID_user', ''),                
            }

        return sample