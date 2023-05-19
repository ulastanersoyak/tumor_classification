# Tumor Classifier

CNN model that can classify given brain MRI images  to 4 classes(glioma tumor, meningioma tumor, no tumor, pituitary tumior)

# Installation

1) Clone the repository:
```bash
git clone https://github.com/UlasTanErsoyak/tumor_classification
cd tumor_classification
```
2) Install the required dependencies specified in the dependencies.txt file.

3) Download the dataset from Kaggle:


    Dataset: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
    Place the dataset files in the appropriate directory.


# Usage

```python run.py --image_size <default 256>  --batch_size <default 32>  --num_workers <default 4> --learning_rate <default 0.0001> --epochs <default 20> --transforms <default 1>```

--image_size: The desired size (in pixels) of the input images for the model. The default value is 256.

--batch_size: The number of images to include in each batch during training. The default value is 32.

--num_workers: The number of subprocesses to use for data loading. The default value is 4.

--learning_rate: The learning rate for the model's optimizer. The default value is 0.0001.

--epochs: The number of training epochs. The default value is 20.

--transforms: Determines whether to apply data augmentation transforms during training. Set it to 1 to enable transforms or 0 to skip them. The default value is 1.







# Contributions 
1)Fork the project

Create your feature branch 

```git checkout -b feature/my-feature```

Commit your changes 

``` git commit -m 'Add some feature' ```

Push to the branch 

``` git push origin feature/my-feature ```

Open a pull request


# Current results


Avarega time to train one epoch on <NVIDIA GeForce GTX 1650> : 64.6149539232 sec/epoch

![Current Graphs](https://github.com/UlasTanErsoyak/tumor_classification/assets/92662728/a8bdf597-d95c-48cf-8188-73a029716927)
