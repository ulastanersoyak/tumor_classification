import torch

def set_device()-> str:
    """
    Set the device to use for running PyTorch operations, either CPU or GPU.

    Returns:
        str: The name of the device, either "cuda" or "cpu".
    """
    if torch.cuda.is_available():
        print(f"device -> {torch.cuda.device(0)}")
        print(f"device name -> {torch.cuda.get_device_name(0)}")
        return "cuda"
    return "cpu"


def calculate_accuracy(true:int,total:int)->float:
    """Calculates the accuracy of the model

        Args:
            true (int): correct prediction count
            total (int): total prediction count
        Returns:
            accuracy (float): % accuracy of the model
    """
    accuracy = (true/total)*100
    return accuracy
