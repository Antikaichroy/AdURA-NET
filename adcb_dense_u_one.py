from densenet import densenet121
from libauc.losses import AUCMLoss, CrossEntropyLoss
from libauc.optimizers import PESG, Adam
#from libauc.datasets import CheXpert
from chexpert_preprocess_u_one import CheXpert
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import torch.nn as nn
from torch.optim import AdamW
import gc
import numpy as np

model = densenet121(pretrained = False, num_classes = 5, last_activation = None, activations = "relu")
#checkpoint = torch.load("/user1/faculty/cvpr/ujjwal/Antik/ce_pretrained_model_hirar_huber.pth")
#model.load_state_dict(checkpoint)

def set_seed(SEED):
  torch.manual_seed(SEED)
  np.random.seed(SEED)
  #random.seed(SEED)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


root = "/user1/faculty/cvpr/ujjwal/Antik/CheXpert/"

train_set = CheXpert(csv_path = root+"train.csv", image_root_path=root, image_size = 320, class_index = -1, use_upsampling=False, shuffle = True, mode ='train', verbose=False)
test_set = CheXpert(csv_path = root+"valid.csv", image_root_path=root, image_size=320, use_upsampling=False, shuffle = False, verbose = False, class_index = -1, mode = "valid")

train_dataloader = DataLoader(dataset = train_set, batch_size = 16, shuffle = True)
test_dataloader = DataLoader(dataset = test_set, batch_size = 16, shuffle = False)

import numpy as np
SEED = 123
lr = 3e-4
weight_decay = 1e-5
batch_size = 16

set_seed(SEED)
model = model.to("cuda:1")

huber = nn.HuberLoss(delta = 1.0)
lambda_orth = 0.05
CELoss = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas = [0.9,0.999])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max=100)
# training
best_val_auc = 0 
for epoch in range(100):
    #torch.cuda.empty_cache()
    #scheduler.step()
    #print("Current lr:", scheduler.get_last_lr()[0])
    for idx, data in enumerate(train_dataloader):
      train_data, train_labels = data
      
      train_data, train_labels  = train_data.to("cuda:1"), train_labels.to("cuda:1")
      multi_dcn , pred = model(train_data)
      offset =multi_dcn['offset']#model.features.MultiScaleDCN_1.ms_dcn_out['offset'][0]
      f = multi_dcn['features'] #model.features.MultiScaleDCN_1.ms_dcn_out['features'][0]
      ce_loss = CELoss(pred, train_labels)
      b,c,h,w = f.shape
      f_flatten = f.view(b,c,-1)  # there will be the batch and the channel and h*w
      # batch matmul
      gram = torch.bmm(f_flatten, f_flatten.transpose(1,2))/(h*w)
      I = torch.eye(c,device = f.device).unsqueeze(0)
      reg_loss = lambda_orth*((gram-I)**2).mean()
      offset_loss = huber(offset, torch.zeros_like(offset))
      #print(ce_loss, reg_loss, offset_loss)
      loss = ce_loss + offset_loss + reg_loss
      
      optimizer.zero_grad()
      loss.backward()
      for name, param in model.named_parameters():
          if name == "offset" and param.grad is not None:
              g = param.grad
              g_norm = torch.norm(g)
              if g_norm>0.1: #exploding
                  param.grad = 0.1*g/(g_norm+1e-6)
                  
              
      optimizer.step()
        
      # validation  
      if idx % 400 == 0:
         model.eval()
         with torch.no_grad():    
              test_pred = []
              test_true = [] 
              for jdx, data in enumerate(test_dataloader):
                  test_data, test_labels = data
                  test_data = test_data.to("cuda:1")
                  _,y_pred = model(test_data)
                  test_pred.append(y_pred.cpu().detach().numpy())
                  test_true.append(test_labels.numpy())
            
              test_true = np.concatenate(test_true)
              test_pred = np.concatenate(test_pred)
              val_auc_mean =  roc_auc_score(test_true, test_pred) 
              model.train()
              del test_true
              del test_pred
              gc.collect()
              torch.cuda.empty_cache()

              if best_val_auc < val_auc_mean:
                 best_val_auc = val_auc_mean
                 torch.save(model.state_dict(), '/user1/faculty/cvpr/ujjwal/Antik/Adaptive_Deformable_Densenet121/adcb_dense121_u_one.pth')

              print('Epoch=%s, BatchID=%s, Val_AUC=%.4f, Best_Val_AUC=%.4f'%(epoch, idx, val_auc_mean, best_val_auc))

    scheduler.step()
    
    print("Epoch LR:", scheduler.get_last_lr()[0])