# pytorch training
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F


# general
import argparse
import numpy as np

# logging
import datetime
time_now  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') 

# other files 
from datasets import BirdSoundDataset, get_datasets
from test_model import BirdCLEFModel, GeM
import config as cfg
# # cmap metrics
import pandas as pd
import sklearn.metrics

#TODO Fix this
device ='cuda' if torch.cuda.is_available() else 'cpu'
loss_print_freq = 50 

parser = argparse.ArgumentParser() 
parser.add_argument('-e', '--epochs', default=10, type=int)
parser.add_argument('-nf', '--num_fold', default=5, type=int)
parser.add_argument('-nc', '--num_classes', default=264, type=int)
parser.add_argument('-tbs', '--train_batch_size', default=16, type=int)
parser.add_argument('-vbs', '--valid_batch_size', default=16, type=int)
parser.add_argument('-sr', '--sample_rate', default=32_000, type=int)
parser.add_argument('-hl', '--hop_length', default=512, type=int)
parser.add_argument('-mt', '--max_time', default=5, type=int)
parser.add_argument('-nm', '--n_mels', default=224, type=int)
parser.add_argument('-nfft', '--n_fft', default=1024, type=int)
parser.add_argument('-s', '--seed', default=0, type=int)
parser.add_argument('-j', '--jobs', default=4, type=int)


#https://www.kaggle.com/code/imvision12/birdclef-2023-efficientnet-training


def loss_fn(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, torch.tensor(labels))
    
#def train(model, data_loader, optimizer, scheduler, device, epoch):
#    model.train()
#
#    running_loss = 0
#    log_n = 0
#    log_loss = 0
#    correct = 0
#    total = 0
#
#    for i, (mels, labels) in enumerate(data_loader):
#        optimizer.zero_grad()
#        mels = mels.to(device)
#        labels = labels.to(device)
#        
#        outputs = model(mels)
#        _, preds = torch.max(outputs, 1)
#        
#        loss = loss_fn(outputs, labels)
#        
#        loss.backward()
#        optimizer.step()
#        
#        
#        if scheduler is not None:
#            scheduler.step()
#            
#        running_loss += loss.item()
#        total += labels.size(0)
#        correct += preds.eq(labels).sum().item()
#        log_loss += loss.item()
#        log_n += 1
#
#        if i % (loss_print_freq) == 0 or i == len(data_loader) - 1:
#            print("Loss:", log_loss, "Accuracy:", correct / total * 100.)
#            log_loss = 0
#            log_n = 0
#            correct = 0
#            total = 0
#
#    return running_loss/len(data_loader)

def train(model, 
          data_loader,
          optimizer,
          scheduler,
          device,
          epoch):
    model.train()

    running_loss = 0
    log_n = 0
    log_loss = 0
    correct = 0
    total = 0
    
    for epoch in range(10):
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
    

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()

            if scheduler is not None:
                scheduler.step()
                
            running_loss += loss.item()
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            log_loss += loss.item()
            log_n += 1
    
            if i % (loss_print_freq) == 0 or i == len(data_loader) - 1:
                print("Loss:", log_loss, "Accuracy:", correct / total * 100.)
                log_loss = 0
                log_n = 0
                correct = 0
                total = 0
    
    return running_loss/len(data_loader)
    
#        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")
def valid(model, data_loader, device, epoch):
    model.eval()
    
    running_loss = 0
    pred = []
    label = []
    
    for i, (mels, labels) in enumerate(data_loader):
        mels = mels.to(device)
        labels = labels.to(device)
        
        outputs = model(mels)
        _, preds = torch.max(outputs, 1)
        
        loss = loss_fn(outputs, labels)
            
        running_loss += loss.item()
        
        pred.extend(preds.view(-1).cpu().detach().numpy())
        label.extend(labels.view(-1).cpu().detach().numpy())
    
    try:
        pd.DataFrame(label).to_csv(f"{time_now}_{epoch}_labels.csv")
        pd.DataFrame(pred).to_csv(f"{time_now}_{epoch}_predictions.csv")
    except:
        print("L your csv(s) died") 
    
    valid_map = sklearn.metrics.average_precision_score(label, pred, average='macro')
    valid_f1 = sklearn.metrics.f1_score(label, pred, average='macro')
    
    return running_loss/len(data_loader), valid_map, valid_f1


def set_seed():
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)

# def padded_cmap(solution, submission, padding_factor=5):
#     solution = solution.drop(['row_id'], axis=1, errors='ignore')
#     submission = submission.drop(['row_id'], axis=1, errors='ignore')
#     new_rows = []
#     for i in range(padding_factor):
#         new_rows.append([1 for i in range(len(solution.columns))])
#     new_rows = pd.DataFrame(new_rows)
#     new_rows.columns = solution.columns
#     padded_solution = pd.concat([solution, new_rows]).reset_index(drop=True).copy()
#     padded_submission = pd.concat([submission, new_rows]).reset_index(drop=True).copy()
#     score = sklearn.metrics.average_precision_score(
#         padded_solution.values,
#         padded_submission.values,
#         average='macro',
#     )
#     return score

if __name__ == '__main__':
    CONFIG = parser.parse_args()
    set_seed()
    print("Loading Model...")
    model = BirdCLEFModel(CONFIG=CONFIG).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-5, T_max=10)
    model = model.to(device)
    print("Model / Optimizer Loading Succesful :P")

    print("Loading Dataset")
    train_dataset, val_dataset = get_datasets(cfg.TRAINING_DATA_DIR/cfg.LABELS_FILE,
                                              cfg.TRAINING_DATA_DIR,
                                              device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        CONFIG.train_batch_size,
        shuffle=True,
        num_workers=CONFIG.jobs
    )
    print("train dataloader len: " + str(len(train_dataloader)))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        CONFIG.valid_batch_size,
        shuffle=False,
        num_workers=CONFIG.jobs
    )
    print("val dataloader len: " + str(len(val_dataloader)))
    
    print("Training")
    for epoch in range(CONFIG.epochs):
        print("Epoch " + str(epoch))

        train_loss = train(
            model, 
            train_dataloader,
            optimizer,
            scheduler,
            device,
            epoch)
        valid_loss, valid_map, valid_f1 = valid(model, val_dataloader, device, epoch)
        print(f"Validation Loss:\t{valid_loss} \n Validation mAP:\t{valid_map} \n Validation F1: \t{valid_f1}" )
        if valid_f1 > best_valid_f1:
            print(f"Validation F1 Improved - {best_valid_f1} ---> {valid_f1}")
            torch.save(model.state_dict(), f'./{time_now}_model_{epoch}.bin')
            print(f"Saved model checkpoint at ./{time_now}_model_{epoch}.bin")
            best_valid_f1 = valid_f1

    print(":o wow")
