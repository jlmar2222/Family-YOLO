import os
import torch
from torch.utils.data import Dataset 
import pandas as pd 
from PIL import Image
import numpy as np


class VocDataset(Dataset):
    def __init__(self, df : pd.DataFrame, img_width: int, img_height: int, S: int, B: int, transforms=None):

        # MEJORAR PARA NO HARDCODEAR Y PASARLO TODO DESDE ARCHIVO YAML

        self.df = df.reset_index(drop=True)
        self.image_names = self.df['file_name'].unique()
        self.transforms = transforms
        self.img_width = img_width
        self.img_height = img_height
        self.S = S
        self.B = B

        classes = sorted(self.df['class'].unique())
        self.C = len(classes)
        self.class2idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        filename = self.image_names[idx]
        image_df = self.df[self.df['file_name'] == filename]

        image_path = image_df.iloc[0]['image_path']

        image = np.array(Image.open(image_path))     
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")  

        bboxes = image_df[['x_min', 'y_min', 'x_max', 'y_max']].values
        class_labels = image_df['class'].values

        if self.transforms:
            # Due to Albumentation library, 
            # we need to pass the image as a dictionary 
            # and get the transformed image back from the dictionary
            transformed = self.transforms(image=image, 
                                            bboxes=bboxes.tolist(), 
                                            labels=class_labels.tolist())
            image = transformed['image']
            bboxes = transformed['bboxes']  
            class_labels = transformed['labels']      

            
        # IGUAL ESTO DECIDIR DESDE EL YAML BUNEO AUNQUE EN REALIDAD ASI ESTA BIEN PORQUE AQUI YA SE HA TRANSFORMADO 
        image_height, image_width = self.img_height, self.img_width

        # ¿Cunto mide cada celda?
        S_h = image_height / self.S
        S_w = image_width / self.S

        # Rellenar YOLO targets --> S * S * (B*5 + C)
        yolo_targets = torch.zeros((self.S, self.S, self.B * 5 + self.C))


        for bbox, class_label in zip(bboxes, class_labels):
     
        
            # ¿Cual es el centro del objeto?
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2

            # ¿Cual es la celda responsable? (la que tiene el centro del objeto)
            i = int(cx / S_w)
            j = int(cy / S_h)
                # Clamp
            i = min(i, self.S - 1)
            j = min(j, self.S - 1)

            # ¿Cual es el valor del centro relativo a la celda? (normalizado a la celda)
            x = (cx / S_w) - i
            y = (cy / S_h) - j

            # ¿Cual es el ancho y alto del bbox normalizados?
            w = (bbox[2] - bbox[0]) / self.img_width
            h = (bbox[3] - bbox[1]) / self.img_height 

            # Solo las celdas 'elegidas' acaban teniendo valroes distintos de cero
            for b in range(self.B):

                yolo_targets[j, i, b * 5 + 0] = x
                yolo_targets[j, i, b * 5 + 1] = y
                yolo_targets[j, i, b * 5 + 2] = w
                yolo_targets[j, i, b * 5 + 3] = h
                yolo_targets[j, i, b * 5 + 4] = 1.0
            
            yolo_targets[j, i, self.B * 5 + self.class2idx[class_label]] = 1.0
          

        sample = {
            'image': image,
            'yolo_targets': yolo_targets,          
            }

        return sample