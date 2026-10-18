# Mango Leaf Disease Detector

A web app that detects mango leaf diseases from a photo.

- Dataset: MangoLeafBD (4000 images, 8 classes)
- Model: MobileNetV2 (transfer learning) with data augmentation
- Validation accuracy: about 99%
- Built with TensorFlow and Streamlit

## Run it

pip install -r requirements.txt
streamlit run app.py

## Classes

Anthracnose, Bacterial Canker, Cutting Weevil, Die Back, Gall Midge, Healthy, Powdery Mildew, Sooty Mould