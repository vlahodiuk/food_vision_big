import torch
import torchvision
from torch import nn

import torch
import torchvision
from torch import nn
def create_effnetb2_model(num_classes: int=3,
                          seed: int=42):
  """Creates an EfficientNetB2 feature extractor model and transforms.

  Args:
      num_classes (int, optional): number of classes in the classifier head.
          Defaults to 3.
      seed (int, optional): random seed value. Defaults to 42.

  Returns:
      model (torch.nn.Module): EffNetB2 feature extractor model.
      transforms (torchvision.transforms): EffNetB2 image transforms.
  """
  # 1. Setup pretrained EffNetB2 weights
  weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  # 2. Get EffNetB2 transforms
  transforms = weights.transforms()
  # 3. Setup pretrained model
  model = torchvision.models.efficientnet_b2(weights=weights) # could also use weights="DEFAULT"
  # 4. Freeze the base layers in the model (this will freeze all layers to begin with)
  for parameter in model.parameters():
    parameter.requires_grad = False

  torch.manual_seed(seed)
  model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1408, out_features=num_classes))

  return model, transforms
