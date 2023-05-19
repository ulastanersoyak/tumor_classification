from typing import List, Tuple
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from model import tumor_classifier
from utils import calculate_accuracy
from torch.optim.lr_scheduler import StepLR
import time
import torch
from tqdm import tqdm

def train_one_epoch(model:nn.Module,train_dataloader:DataLoader,loss_fn:nn.Module,optimizer:optim.Optimizer,device:str)-> Tuple[float,float]:
    """
    Trains the given `model` for one epoch on the provided `train_dataloader` using the specified `loss_fn`
    and `optimizer`. Calculates the training accuracy and loss for the entire training dataset.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_dataloader (DataLoader): The training data.
        loss_fn (nn.Module): The loss function to be used during training.
        optimizer (optim.Optimizer): The optimizer to be used during training.
        device (str): The device to be used for training (e.g. "cuda:0" or "cpu").

    Returns:
        A tuple containing the average training loss and accuracy for the entire training dataset.
    """
    model.to(device)
    model.train()
    train_acc = 0
    train_loss=0    
    for imgs,labels in train_dataloader:
        imgs,labels = imgs.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(labels)
        train_acc += (outputs.argmax(dim=1) == labels).sum().item()

    total_samples = len(train_dataloader.dataset)
    train_loss /= total_samples
    train_acc = calculate_accuracy(train_acc, total_samples)
    return train_loss,train_acc

def train(model:nn.Module,train_dataloader:DataLoader,loss_fn:nn.Module,optimizer:optim.Optimizer,device:str,epochs:int) -> Tuple[List[float], List[float]]:
    """
    Trains the given `model` on the provided `train_dataloader` for the specified number of `epochs`, using the
    specified `loss_fn`, `optimizer`, and `learning_rate`. Calculates the training accuracy and loss for each epoch.

    Args:
        model (nn.Module): The PyTorch model to be trained.
        train_dataloader (DataLoader): The training data.
        loss_fn (nn.Module): The loss function to be used during training.
        optimizer (optim.Optimizer): The optimizer to be used during training.
        device (str): The device to be used for training (e.g. "cuda:0" or "cpu").
        epochs (int): The number of epochs to train the model.

    Returns:
        A tuple containing three lists: the average training loss for each epoch, the training accuracy for each epoch, and the time each epoch took.
    """
    total_train_loss=[]
    total_train_accuracy=[]
    total_time=[]
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    train_start = time.time()
    print("TRAINING STARTED")
    for i in tqdm(range(epochs), desc="Epochs"):
        start = time.time()
        train_loss,train_acc = train_one_epoch(model=model,
                                               train_dataloader=train_dataloader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               device=device)
        total_train_loss.append(train_loss)
        total_train_accuracy.append(train_acc)
        scheduler.step()
        end = time.time()
        total= end-start
        total_time.append(total)
        print("----------------------------------------------------------------------------------------------------------------------------------")
        print("|                                                                                                                                |")
        if(i>9):
            print(f"|    Epoch {i+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: %{train_acc:.4f} | Train time: {total:.2f}second(s)                                         |")
        else:
            print(f"|    Epoch {i+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: %{train_acc:.4f} | Train time: {total:.2f}second(s)                                          |")
        print("|                                                                                                                                |")
        print("----------------------------------------------------------------------------------------------------------------------------------")
        if(i%10==0):
            torch.save(model.state_dict(), "tumor_classifier.pth")
    torch.save(model.state_dict(), "tumor_classifier.pth")
    train_end = time.time()
    print(f"TRAINING FINISHED | Took {train_end-train_start}second(s)")
    return total_train_loss, total_train_accuracy,total_time

def test_one_epoch(model:nn.Module,test_dataloader:DataLoader,loss_fn:nn.Module,device:str)-> Tuple[float,float]:
    model.to(device)
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for imgs,labels in test_dataloader:
            imgs,labels = imgs.to(device),labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            test_loss += loss.item() * len(labels)
            test_acc += (outputs.argmax(dim=1) == labels).sum().item()

        total_samples = len(test_dataloader.dataset)
        test_loss /= total_samples
        test_acc = calculate_accuracy(test_acc, total_samples)
        return test_loss,test_acc

def test(model:nn.Module,test_dataloader:DataLoader,loss_fn:nn.Module,device:str,epochs:int)-> Tuple[List[float], List[float]]:
    """
    Performs testing on a trained model using the provided test dataloader.

    Args:
        model (nn.Module): The trained model to be tested.
        test_dataloader (DataLoader): The dataloader containing the test dataset.
        loss_fn (nn.Module): The loss function used for calculating the test loss.
        device (str): The device (e.g., 'cpu' or 'cuda') on which the testing will be performed.
        epochs (int): The number of epochs to run the testing.

    Returns:
        Tuple[List[float], List[float], List[float]]: A tuple containing the lists of test losses, test accuracies, and the time taken for each epoch.

    """
    total_test_loss=[]
    total_test_accuracy=[]
    total_time=[]
    test_start = time.time()
    print("TESTING STARTED")
    for i in tqdm(range(epochs), desc="Epochs"):
        start = time.time()
        test_loss,test_acc = test_one_epoch(model=model,
                                               test_dataloader=test_dataloader,
                                               loss_fn=loss_fn,
                                               device=device)
        total_test_loss.append(test_loss)
        total_test_accuracy.append(test_acc)
        end = time.time()
        total= end-start
        total_time.append(total)
        print("----------------------------------------------------------------------------------------------------------------------------------")
        print("|                                                                                                                                |")
        if(i>9):
            print(f"|    Epoch {i+1}/{epochs} | Test Loss: {test_loss:.4f} | Test Acc: %{test_acc:.4f} | Test time: {total:.2f}second(s)                                            |")
        else:
            print(f"|    Epoch {i+1}/{epochs} | Test Loss: {test_loss:.4f} | Test Acc: %{test_acc:.4f} | Test time: {total:.2f}second(s)                                             |")
        print("|                                                                                                                                |")
        print("----------------------------------------------------------------------------------------------------------------------------------")
    torch.save(model.state_dict(), "tumor_classifier.pth")
    test_end = time.time()
    print(f"TESTING FINISHED | Took {test_end-test_start}second(s)")
    return total_test_loss, total_test_accuracy,total_time



def test_and_train(model:nn.Module,train_dataloader:DataLoader,test_dataloader:DataLoader,loss_fn:nn.Module,optimizer:optim.Optimizer,device:str,epochs:int) ->Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains and tests a given model for a certain number of epochs, using the specified training and test dataloaders,
    loss function, optimizer, and device.

    Args:
        model (nn.Module): The neural network model to train and test.
        train_dataloader (DataLoader): The dataloader containing the training dataset.
        test_dataloader (DataLoader): The dataloader containing the test dataset.
        loss_fn (nn.Module): The loss function to use during training and testing.
        optimizer (optim.Optimizer): The optimizer to use during training.
        device (str): The device to use for training and testing (e.g., 'cpu' or 'cuda').
        epochs (int): The number of epochs to train for.

    Returns:
        Tuple of six lists:
        - total_train_loss (List[float]): The training loss for each epoch.
        - total_train_accuracy (List[float]): The training accuracy for each epoch.
        - train_time (List[float]): The time taken for each epoch of training.
        - total_test_loss (List[float]): The test loss for each epoch.
        - total_test_accuracy (List[float]): The test accuracy for each epoch.
        - test_time (List[float]): The time taken for each epoch of testing.
    """
    test_train_start = time.time()
    print("TRAINIG AND TESTING STARTED")
    total_train_loss=[]
    total_train_accuracy=[]
    train_time=[]
    scheduler = StepLR(optimizer, step_size=10, gamma=0.6)
    
    total_test_loss=[]
    total_test_accuracy=[]
    test_time=[]
    min_loss = 99999

    for i in tqdm(range(epochs), desc="Epochs"):
        ### TRAINING PART ###
        start = time.time()
        train_loss,train_acc = train_one_epoch(model=model,
                                               train_dataloader=train_dataloader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               device=device)
        total_train_loss.append(train_loss)
        total_train_accuracy.append(train_acc)
        scheduler.step()
        end = time.time()
        total= end-start
        train_time.append(total)
        print("\n----------------------------------------------------------------------------------------------------------------------------------")
        print("|                                                                                                                                |")
        if(i>8):
            print(f"|    Epoch {i+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: %{train_acc:.4f} | Train time: {total:.2f}second(s)                                         |")
        else:
            print(f"|    Epoch {i+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: %{train_acc:.4f} | Train time: {total:.2f}second(s)                                          |")
        print("|                                                                                                                                |")

        ### TESTING PART ###
        start = time.time()
        test_loss,test_acc = test_one_epoch(model=model,
                                               test_dataloader=test_dataloader,
                                               loss_fn=loss_fn,
                                               device=device)
        total_test_loss.append(test_loss)
        total_test_accuracy.append(test_acc)
        end = time.time()
        total= end-start
        test_time.append(total)
        print("|                                                                                                                                |")
        if(i>8):
            print(f"|    Epoch {i+1}/{epochs} | Test Loss: {test_loss:.4f} | Test Acc: %{test_acc:.4f} | Test time: {total:.2f}second(s)                                            |")
        else:
            print(f"|    Epoch {i+1}/{epochs} | Test Loss: {test_loss:.4f} | Test Acc: %{test_acc:.4f} | Test time: {total:.2f}second(s)                                             |")
        print("|                                                                                                                                |")
        print("----------------------------------------------------------------------------------------------------------------------------------")
        if(test_loss<min_loss):
            torch.save(model.state_dict(), "tumor_classifier.pth")
            print(f"Saved a new best model! Previous loss was: {min_loss} new loss is: {test_loss}")
            min_loss=test_loss

    test_train_end = time.time()
    print(f"TRAINING ANG TESTING FINISHED | Took {test_train_end-test_train_start}second(s)")
    return total_train_loss, total_train_accuracy, train_time, total_test_loss, total_test_accuracy, test_time

def evaluate(model_path,target):
        img = target[0]
        label = target[0]
        
        model = tumor_classifier()
        model.load_state_dict(torch.load(model_path))
        prediction = model(img)
        return f"predicted {prediction} /|\ target: {label}"