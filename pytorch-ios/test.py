import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os

test_data_dir = f'data/test'
test_data_files = os.listdir(test_data_dir)

PATH = 'model.pt'
model = torch.jit.load(PATH)


def load_and_show(i):
    img = Image.open(f'{test_data_dir}/{test_data_files[i]}').convert("RGB")
    # RGB is important because we don't want RGBA with 4 channels!
    plt.imshow(img)
    plt.show()
    return img


# See https://pytorch.org/hub/pytorch_vision_mobilenet_v2/ for necessary mobilenet preprocessing
def transform_image(pillow_img):
    img = transforms.functional.resize(pillow_img, [224, 224])
    img = transforms.functional.to_tensor(img)
    img = transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    minibatch = torch.stack([img])
    return minibatch


def predict(input_model, x):
    softMax = nn.Softmax(dim = 1)
    outputs = input_model(x)
    probs = softMax(outputs)
    _, preds = torch.max(outputs, 1)
    return probs, preds.item()


IMG_NUM = 2
img_pillow = load_and_show(IMG_NUM)
img_tensor = transform_image(img_pillow)
probabilities, label = predict(model, img_tensor)

# label 0 = cat, label 1 = dog
class_names = ["cat", "dog"]
print(probabilities)
print(class_names[label])