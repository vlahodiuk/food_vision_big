
### 1. Imports and class names setup ###
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

device = "cpu"
# Setup class names
with open(foodvision_big_class_names_path, "r") as f:
  class_names = [food.strip() for food in f.readlines()]

### 2. Model and transforms preparation ###
# Create Effnetb2 model
effnetb2_model, effnetb2_transforms = create_effnetb2_model(
    num_classes=101
)
# Load sawed weight
effnetb2_model.load_state_dict(
    torch.load(
        f="09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth",
        map_location=torch.device("cpu")
    )
)

### 3. Predict function ###
def predict(img) -> Tuple[Dict, float]:
  # Start a timer
  start_time = timer()
  # Transform input image for EffNetB2
  img = effnetb2_transforms(img).unsqueeze(0).to(device)
  # Put model into veval mode
  effnetb2_model.eval()
  with torch.inference_mode():
    pred_probs = torch.softmax(effnetb2_model(img), dim=1)

  # Create a prediction label and pred probability
  pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

  # Calculate pred time and return pred dict and pred time
  pred_time = round(timer()-start_time, 5)
  return pred_labels_and_probs, pred_time

### 4. Gradio app ###
title = "FoodVision Big 🍔👁"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food into [101 different classes](https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/food101_class_names.txt)."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

example_list = [["examples/"+example] for example in os.listdir("examples")]

# Create a Gradion demo
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                            gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)
# Launch the demo!
demo.launch()
