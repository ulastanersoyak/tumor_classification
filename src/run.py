import torch
from data_setup import tumor_dataset,create_dataloaders
from model import tumor_classifier
from engine import train,test,evaluate,test_and_train
import argparse
from torch import nn,optim
import matplotlib.pyplot as plt
from utils import set_device
import os
import torchvision.transforms as transforms

PATH = "tumor_classifier.pth"
IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
EPOCHS=20
TRANSFORMS = 1
parser = argparse.ArgumentParser(description='hyperparameters')

parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                    help='batch size for dataloaders')

parser.add_argument('--num_workers', type=int, default=NUM_WORKERS,
                    help='number of workers for dataloaders')

parser.add_argument('--image_size', type=tuple, default=IMAGE_SIZE,
                    help='resolution for images')

parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                    help='learning rate for optimizer')

parser.add_argument('--epochs', type=int, default=EPOCHS,
                    help='number of iteration over dataset')

parser.add_argument('--transforms', type=int, default=EPOCHS,
                    help='number of iteration over dataset')

args = parser.parse_args()

if __name__ == "__main__":
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    transform_int = args.transforms

    train_path= "data\Training"
    test_path= "data\Testing"

    if(transform_int == 1):
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()])

        train_dataset= tumor_dataset(root_dir=train_path,image_size=(image_size,image_size),transform=transform)

    else:
        train_dataset= tumor_dataset(root_dir=train_path,image_size=(image_size,image_size))
    
    test_dataset = tumor_dataset(root_dir=test_path,image_size=(image_size,image_size),)

    train_dataloader,test_dataloader = create_dataloaders(train_dataset=train_dataset,
                                                          test_dataset=test_dataset,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          shuffle=True)
    





    # evaluate("tumor_classifier.pth",test_dataset[0])

    model = torch.nn.Module
    if os.path.isfile(PATH):
        print('found trained model.')
        model = tumor_classifier()
        model.load_state_dict(torch.load(PATH))
    else:
        model = tumor_classifier()
        print('created model')


    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(),lr=learning_rate,weight_decay=0.005,momentum=0.9)
    device = set_device()

    # train_loss,train_acc,train_time= train(model=model,
    #                                        train_dataloader=train_dataloader,
    #                                        loss_fn=loss_fn,
    #                                        optimizer=optimizer,
    #                                        epochs=epochs,
    #                                        device=device)
    
    # test_loss,test_acc,test_time = test(model=model,
    #                                     test_dataloader=test_dataloader,
    #                                     loss_fn=loss_fn,
    #                                     device=device,
    #                                     epochs=epochs)
    
    train_loss, train_acc, train_time, test_loss, test_acc, test_time = test_and_train(model=model,
                                                                                       train_dataloader=train_dataloader,
                                                                                       test_dataloader=test_dataloader,
                                                                                       loss_fn=loss_fn,
                                                                                       optimizer=optimizer,
                                                                                       device=device,
                                                                                       epochs=epochs)

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(train_time, label='Train Time')
    plt.plot(test_loss, label='Test Loss')
    plt.plot(test_acc, label='Test Accuracy')
    plt.plot(test_time, label='Test Time')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()