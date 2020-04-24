import io
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
# import json

# Model import
def get_model():
    model = models.densenet121(pretrained = True)
    fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 5),
        nn.LogSoftmax(dim = 1)
    )
    model.classifier = fc
    model.load_state_dict(torch.load('Smap_predictor.pt', map_location=torch.device('cpu')))
    model.eval()
    return model

# Transform image
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# # Json class encoder
# class JsonEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(MyEncoder, self).default(obj)