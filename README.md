# 🍕 Food Vision Big (EfficientNet-B2)

Food image classification project built with **PyTorch** using a **pretrained EfficientNet-B2** model.  
The model is fine-tuned on the **Food-101 dataset** and packaged as a simple demo, ready for deployment on **Hugging Face Spaces**.

This repository focuses on:
- Transfer learning with modern CNN architectures
- Clean inference pipeline
- Practical deployment readiness

---

## 🚀 Features

- Pretrained **EfficientNet-B2**
- Fine-tuned on **Food-101**
- PyTorch inference pipeline
- Lightweight demo app
- Hugging Face Spaces compatible
- Simple, readable project structure

---

## 🧠 Model Overview

- **Architecture:** EfficientNet-B2  
- **Framework:** PyTorch  
- **Dataset:** Food-101  
- **Training strategy:** Transfer learning  
- **Checkpoint:** `09_pretrained_effnetb2_food101_20_percent.pth`

The classifier head is replaced and fine-tuned for food category prediction.

---

## 📁 Project Structure

```
.
├── app.py
├── model.py
├── class_names.txt
├── examples/
│   ├── pizza.jpg
│   ├── sushi.jpg
│   └── steak.jpg
├── requirements.txt
├── README.md
└── LICENSE
