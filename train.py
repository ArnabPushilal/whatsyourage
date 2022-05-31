
import torch
import numpy as np
import time
from dataloader import *
from model import VGGnet
import torch.optim as optim
import torch.nn as nn

def train(trainloader,valloader,model,optimizer,criterion,epochs=10,device='cuda',return_val=False,print_=True):
         
     model=model.to(device) 
     val_best=np.inf
     best_model=None
        
     for epoch in range(epochs):

        train_loss = []
        val_loss=[]
        time_epoch=time.time() 
        for i, batch_data in enumerate(trainloader, 1):
     
            inputs, age = batch_data  
            inputs = inputs.float().to(device)
            age = age.float().to(device)
            optimizer.zero_grad()
            age_pred = model(inputs)
            age_pred=age_pred.float()        
            loss = torch.sqrt(criterion(torch.squeeze(age_pred),age))
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        time_epoch_vl=time.time() 
        if print_:
            print('----------------------------------------------------------------------------------')
            print(f"Epoch: {epoch+1} Time taken : {round(time_epoch_vl-time_epoch,3)} seconds")
            print("-----------------------Training Metrics-------------------------------------------")
            print("Loss: ",round(np.mean(train_loss),3))
    
    
        for i, batch_data in enumerate(valloader, 1):
        
         with torch.no_grad():
     
            inputs, age = batch_data
            
            inputs = inputs.float().to(device)
            
        
            age = age.float().to(device)
            optimizer.zero_grad()
            age_pred = model(inputs)
            age_pred=age_pred.float()
            loss = torch.sqrt(criterion(torch.squeeze(age_pred),age))
            val_loss.append(loss.item())
            
        
        if print_:

            print("-----------------------Validtion Metrics-------------------------------------------")
            print("Loss: ",round(np.mean(val_loss),3))
        
        ### Saves models based on validation loss & if epoch is >3
        if np.mean(val_loss) < val_best and epoch > 3:
            
            val_best=np.mean(val_loss)
            best_model=model
            if print_:
                print("Model saved")
            
     if return_val:
        
      return best_model,val_best
     else:
      return best_model



if __name__=="__main__":
    dataset_folder_name = 'UTKFace'
  
    df = parse_dataset(dataset_folder_name)
    
    
    data_generator = UtkFaceDataGenerator(df)
    train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()
    
    traindata=new_data(df,train_idx)
    valdata=new_data(df,valid_idx)
    testdata=new_data(df,test_idx)

    trainloader = torch.utils.data.DataLoader( 
    traindata,
    batch_size=12,
    shuffle=True)

    valloader = torch.utils.data.DataLoader(
        valdata,
        batch_size=12,
        shuffle=True)
    #Setting device to cuda for GPU support
    device='cuda'
    model=VGGnet().to(device)

    #Initializing optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  

    #Loss func
    criterion = nn.MSELoss()

    best_model=train(trainloader,valloader,model,optimizer,criterion,epochs=10)
    torch.save(best_model.state_dict(), 'new_model.pt')


    
    


