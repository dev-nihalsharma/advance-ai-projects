import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import load_checkpoint, print_examples, save_checkpoint
from model import CNNtoRNN
from get_loader import get_loader

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((299,299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         ]
    )

   
    train_loader, dataset = get_loader(
        root_folder='flicker8k/Images/',
        annotation_file= 'flicker8k/captions.csv',
        transform=transform,
        batch_size=32,
        num_workers=2
    )
    
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    load_model = False
    save_model = True

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs= 10

    # for Tensorboard
    writer = SummaryWriter('runs/flicker')
    step= 0

    # initializing model, lost function and optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size,num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load('model_checkpoint.pth.tar'), model, optimizer)

    model.train()

    for _ in range(num_epochs):
        print_examples(model,device,dataset)
        if save_model:
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
            }
            save_checkpoint(checkpoint)

        for imgs,captions in  train_loader:
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs  = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            writer.add_scalar('Training Loss', loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

if __name__ == '__main__':
    train()
