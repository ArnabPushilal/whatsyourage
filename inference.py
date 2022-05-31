
import torch
import numpy as np
from dataloader import *
from model import VGGnet

def test_model(image,model,device='cpu'):
    
    model.to(device)
    inputs=image
    inputs = inputs.float().to(device)
    age_pred = model(inputs)
    age_pred=age_pred.float()
    return round(age_pred.item())
            

def preprocess_image(img_path):
        """
        Single Image Preprocessing
        """
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        im = np.array(im) / 255.0
        im=im.T

        im=torch.tensor(im)
        
        im=im[None,:,:,:]

    
        return im
            

if __name__=="__main__":

    im=preprocess_image('ucl-logo_5.png')
    model=VGGnet()  
    model.load_state_dict(torch.load('new_model.pt'))
    age=test_model(im,model)
    print(age)


   

