import unittest
import torch
from data_setup import create_dataloaders,tumor_dataset
class TestTumorDataset(unittest.TestCase):
    
    def test_dataset_creation(self):
        # Test if the dataset is created properly
        dataset = tumor_dataset('data/Training', (256, 256))
        self.assertEqual(len(dataset), 2870)
        img, label = dataset[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertIsInstance(label, int)
        
    def test_visualize_case(self):
        # Test if visualize_case function works
        dataset = tumor_dataset('data/Training', (256, 256))
        self.assertIsNone(dataset.visualize_case(0))

    def test_create_dataloaders(self):
        # Test if create_dataloaders function works
        train_dataset = tumor_dataset('data/Training', (256, 256))
        test_dataset = tumor_dataset('data/Testing', (256, 256))
        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, 32, 4, True)
        self.assertIsInstance(train_loader, torch.utils.data.DataLoader)
        self.assertIsInstance(test_loader, torch.utils.data.DataLoader)
        self.assertEqual(len(train_loader.dataset), 2870)
        self.assertEqual(len(test_loader.dataset), 394)

tests = TestTumorDataset()
tests.test_dataset_creation()
tests.test_visualize_case()
tests.test_create_dataloaders()
print("ALL TESTS PASSED!")