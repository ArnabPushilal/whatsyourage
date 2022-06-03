
import torch
import numpy as np
from dataloader import *
from model import VGGnet

def test_model(image,model,device='cpu'):

    """
    Tests model for one image
    
    """
    
    model.to(device)
    inputs=image
    inputs = inputs.float().to(device)
    age_pred = model(inputs)
    age_pred=age_pred.float()
    return round(age_pred.item())
            

def preprocess_image(img):
        """
        @params:
        img(cv2): Numpy array img in BGR format

        Returns
        im(Tensor): Pytorch tensor
        
        """
        im= Image.fromarray(img)
        #im=Image.open(img)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        im = np.array(im) / 255.0
        im=im.T

        im=torch.tensor(im)
        
        im=im[None,:,:,:]

    
        return im
            


   

