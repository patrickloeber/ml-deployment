# PyTorch iOS
Build a Cats / Dogs classifier, train and optimize it for mobile, and then save it.

## 0. Prepare Data
Data is used from here:

[https://www.kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats)

For PyTorch the images need to be arranged in the folder structure listed below. Download the data from Kaggle, extract it, and re-organize it. The test folder can be used as is, but train and val needs to be further split into cat and dog folder. Move the images into the corresponding folders, and take about 20% from the training folder for validation.

- data/
    - train/
        - cat/
            - cat1.jpg
            - cat2.jpg...
        - dog/
            - dog1.jpg
            - dog2.jpg...
    - val/
        - cat/
            - cat10000.jpg
            - cat10001.jpg...
        - dog/
            - dog10000.jpg
            - dog10001.jpg...
    - test/
        - img.jpg
        ...

## 1. Train the mode
Run
```
python train.py
```

It uses a transfer learning approach and takes a pretrained Mobilenet.
Finetuning is applied with the Cats/Dogs data, and then the model is optimized for mobile using Torchscript and saved. This should dump the file `model.pt`.

#### Resoures
- [Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Mobilenet](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)
- [PyTorch iOS](https://pytorch.org/mobile/ios/)
- [Torchscript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)

## 2. Test the loaded model

Run
```
python test.py
```

Have a look at [test.py](test.py) to see how the model is loaded, how new images have to be preprocessed, and how the model output is interpreted as cat or dog.

## 3. TODO: Build iOS App
Important: Before a new image is fed to the model, the image must be resized and normalized. Compare with the transformation steps in [test.py](test.py)). Also, the image must be in RGB format (no alpha channel), so that the input tensor for the model has the shape [1, 3, 224, 224].

```python
def transform_image(pillow_img):
    transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
```

Note: Normalization values and image size are based on the original [Mobilenet preprocessing](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/) steps: 