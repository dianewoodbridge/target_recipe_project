from PIL import Image
import requests
from io import BytesIO
import numpy as np

class Mapper():
    def __init__(self, df):
        self.df = df
    
    def get_image(self, tcin): 
        url = self.df[self.df['tcin'] == tcin]['image'].iloc[0]
        img = np.nan
        if not isinstance(url, float):
            img = self.get_image_url(url)
        return img

    def get_image_url(self, url):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img

    def get_images_urls(self, urls):
        return [self.get_image_url(url) for url in urls]
    
    def get_image_list(self, tcin_list):
        return [self.get_image(tcin) for tcin in tcin_list]
    
    def get_images(self, tcin): 
        urls = self.df[self.df['tcin'] == tcin]['images'].iloc[0]
        images = []
        if urls:
            for url in urls:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                images.append(img)
        return images
    
    def get_images_list(self, tcin_list): 
        return [self.get_images(tcin) for tcin in tcin_list]
    
    def get_title(self, tcin): 
        return self.df[self.df['tcin'] == tcin]['title'].iloc[0]
    
    def get_title_list(self, tcin_list):
        return [self.get_title(tcin) for tcin in tcin_list]

    def get_column_value(self, tcin, column): 
        return self.df[self.df['tcin'] == tcin][column].iloc[0]
    
    def get_column_list(self, tcin_list, column):
        return [self.get_column_value(tcin, column) for tcin in tcin_list]        