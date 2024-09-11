from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from prettytable import PrettyTable
import my_net as mn
import my_sampler as ms

#------------------------------------------
batch_size = 256
test_batch_size = 256
learning_rate = 0.001
scheduler_step_size = 50
scheduler_gamma = 0.5
use_scheduling = False

w_decay = 1e-7
saved_epoch = 0
epochs = 500
FIRST_RUN = True
saved_path = "model-L8-500.pth"
path = "model-L8-500.pth"

classes = [0, 1]


def train_epoch(model, device, dataloader, optimizer, loss_fn, batch_size):
    train_loss = 0.0
    train_accuracy = 0
    f1_metric = 0.0
    
    model.train()
        
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        
        # Compute loss
        model.zero_grad(set_to_none = True)
        outputs = model(X)
        loss = loss_fn(outputs, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
		
        train_loss += loss.item()

        #compute predictions for train data
        correct_predictions = 0
        total = 0
        y_pred = [0] * batch_size
                
        # the class with the highest energy is what we choose as prediction
        _, pred_idx = torch.max(outputs.data, dim=1)
        
        total += y.size(0)
        for j in range(batch_size):
            correct_predictions += (classes[pred_idx[j]] == y[j]).item()
            y_pred[j] = classes[pred_idx[j]]
        
        train_accuracy += correct_predictions/total

        f1_metric += f1_score(y.cpu(), y_pred)
        
    return train_loss/len(dataloader), train_accuracy/len(dataloader), f1_metric/len(dataloader)


def validate_epoch(model, device, dataloader, batch_size):
    model.eval()
        
    val_accuracy = 0
    f1_metric = 0.0
    
    #compute predictions for validation data 
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
                
        with torch.no_grad():
             outputs = model(X)
        
        #compute predictions for test data
        correct_predictions = 0
        total = 0
        y_pred = [0] * batch_size
                
        # the class with the highest energy is what we choose as prediction
        _, pred_idx = torch.max(outputs.data, dim=1)
        
        total += y.size(0)
        for j in range(batch_size):
            correct_predictions += (classes[pred_idx[j]] == y[j]).item()
            y_pred[j] = classes[pred_idx[j]]
        
        val_accuracy += correct_predictions/total
        f1_metric += f1_score(y.cpu(), y_pred)
                
    return val_accuracy/len(dataloader), f1_metric/len(dataloader)
            

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


# Create data loaders.
train_dataset = ms.NLPDataset(data_file='/home/mkr/lama3/data/train.json', embedder_on=True)

m=len(train_dataset)

#random_split randomly split a dataset into non-overlapping new datasets of given lengths
train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])

# The dataloaders handle shuffling, batching, etc...
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True, drop_last = True)
val_loader = DataLoader(val_data, batch_size=test_batch_size, drop_last = True)

### Set the random seed for reproducible results
torch.manual_seed(0)

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

#init the model
model = mn.nlp_nn()
model.to(device)
count_parameters(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

if use_scheduling:
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = scheduler_step_size, gamma = scheduler_gamma)

if not FIRST_RUN:
	checkpoint = torch.load(saved_path, map_location='cpu')
	model = model.to(device)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	scheduler.load_state_dict(checkpoint['lr'])
	saved_epoch = checkpoint['epochs']
	
else:
	saved_epoch = 0

print('Lets get ready for the rumble with', epochs, end = ' ' )
print('epochs!')

for t in range(saved_epoch, epochs):
    #print(' ')
    #print(f"Epoch {t+1}\n-------------------------------")
    train_loss, mean_accuracy, train_f1 = train_epoch(model=model, device=device, dataloader = train_loader, optimizer=optimizer, loss_fn=loss_fn, batch_size=batch_size)
    val_accuracy, val_f1 = validate_epoch(model=model, device=device, dataloader=val_loader, batch_size=test_batch_size)
    print('\n EPOCH {}/{} \t train loss {:.7f} \t train acc {:.7f} \t train f1 {:.7f} \t val acc {:.7f} \t val f1 {:.7f}'.format(t + 1, epochs, train_loss, mean_accuracy, train_f1, val_accuracy, val_f1))

    #scheduler.step

print("Done!")

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epochs
            }, path)

print(f"Saved PyTorch Model State to  {path}")