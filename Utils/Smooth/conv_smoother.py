import numpy as np
import torch
from torch import nn

class LAIDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # select sample
        X = self.data[index]
        y = self.labels[index]

        return X, y

class CONV(nn.Module):
    
    def __init__(self, num_anc, sws, lr=0.001):
        super(CONV, self).__init__()
        self.sws = sws
        self.num_anc = num_anc
        
        self.smoothNet = self._get_conv_smoother()
        self.drop = nn.Dropout(0.5)
        self.drop_cols = nn.Dropout(0.5)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = 0.001)
        
    def _get_conv_smoother(self):
        layers = []
        layers.append(nn.Conv1d(self.num_anc,self.num_anc,self.sws,padding=(self.sws-1)//2,padding_mode="reflection"))

        return nn.Sequential(*layers)
        
    def forward_tensors(self,base_out):
        """
        Inputs:
        snps: batch of actual snps - encodings could be either 0/1 or -1/1. Shape: (N,chromosome_length)

        Outputs:
        base_out: Probabilities of each class from base network.
        smooth_out: Probabilities of each class from smooth network.

        """
        # computes probabilities.
        smooth_out = self.smoothNet(base_out)
        smooth_out = nn.Softmax(dim=1)(smooth_out)
            
        return smooth_out
    
    
    def forward(self,base_out,labels):
        """
        Inputs:
        snps: batch of actual snps - encodings could be either 0/1 or -1/1. Shape: (N,chromosome_length)
        labels: batch of ground truth labels - Shape: (N, number_of_windows)
        
        Outputs:
        smooth_out: Probabilities of each class from smooth network.
        loss: overall loss for that mini-batch.

        Note: Assumes coordinates are given and are not None. 
        """

        # computes losses.
        smooth_out = self.forward_tensors(base_out)
        loss = nn.NLLLoss()(torch.log(smooth_out + 1e-8),labels)
        
        return smooth_out, loss

    def predict_proba_torch(self, base_out):
        proba_torch = self.forward_tensors(base_out)
        return proba_torch

    def predict_torch(self, base_out):
        smooth_out = self.forward_tensors(base_out)
        preds = torch.argmax(smooth_out,dim=1)
        return preds
    
    def validate(self,base_out,labels):
        """
        Inputs:
        snps: batch of actual snps - encodings could be either 0/1 or -1/1. Shape: (N,chromosome_length)
        labels: batch of ground truth labels - Shape: (N, number_of_windows)
        coordinates: Either batch of ground truth n-vector coordinates or empty tensor - Shape: (N, 3, number_of_windows)
        
        Outputs:
        metrics: list of metrics. First one is always a classification metric, remaining are regression metrics.

        Note: Assumes coordinates are given and are not None. 
        """       
        preds = self.predict_torch(base_out)
        accuracy = float(torch.mean((preds==labels),dtype=torch.float))
        
        return accuracy
    
    # XGMix interface
    def fit(self, X, y, max_ep=250, val_every=50, verbose=False):

        X_torch, y_torch = as_torch_tensor(X, y)
        generator = get_data_generator(X_torch, y_torch)

        for ep in range(1,max_ep+1):
            
            running_loss = 0.0
            for tr,trl in generator:
                self.optimizer.zero_grad()
                self.train()
                outsss,loss = self(tr.float(),trl)
                running_loss += loss
                loss.backward()
                self.optimizer.step()
            
            if verbose:
                print("Loss at iteration {}: {}".format(ep,running_loss.data.cpu().numpy()/len(generator)))
            
            if verbose and ep % val_every == 0:
                
                self.eval()
                
                vals_b, lens_b = [], []
                tval_iter = 0
                for v, vall in generator:
                    tmp_accr = self.validate(v.float(), vall)
                    vals_b.append(tmp_accr)
                    lens_b.append(len(v))
                    tval_iter += 1
                    if tval_iter == 3:
                        break
                vals_b = np.asarray(vals_b)
                lens_b = np.asarray(lens_b)
                
                new_accr = np.sum(vals_b * lens_b) / np.sum(lens_b)
                
                print("Sample Training accuracy after {} iterations: {}".format(ep,new_accr))
            
    def predict(self, X):
        X_torch = as_torch_tensor(X)
        y_pred_torch = self.predict_torch(X_torch)
        y_pred = np.array(y_pred_torch, dtype=int)
        return y_pred

    def predict_proba(self, X):
        X_torch = as_torch_tensor(X)
        proba_torch = self.predict_proba_torch(X_torch)
        proba_np = proba_torch.detach().numpy()
        proba = np.swapaxes(proba_np,1,2)
        return proba

def as_torch_tensor(base_proba, labels=None):

    base_proba = np.copy(base_proba)
    B = np.transpose(base_proba,[0,2,1])
    B_torch = torch.tensor(B,dtype=torch.float)

    if labels is not None:
        y = np.copy(labels)
        y_torch  = torch.tensor(y,dtype=torch.long)
        return B_torch, y_torch

    return B_torch

def get_data_generator(B_torch, y_torch, shuffle=True):

    dataset = LAIDataset(B_torch, y_torch)
    generator = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle = shuffle, num_workers = 0)

    return generator
