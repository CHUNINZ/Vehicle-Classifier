# Vehicle Classifier — ResNet and Residual Blocks (MATLAB)

Classifies images as **sedan, SUV, motorcycle, truck, or van** using transfer learning with **ResNet-18**, a convolutional neural network built from residual blocks. The project includes MATLAB training code and a GUI app for predictions.

## Files
- `train_vehicle_resnet.m` — loads your images, fine-tunes ResNet-18, saves `vehicleResNetModel.mat`
- `VehicleClassifierApp.m` — GUI app: upload an image → get predicted vehicle type + confidence chart
- `vehicle_images/` — put your training images here (already scaffolded with 5 class folders)

## 1. Install requirements (one-time)
In MATLAB: **Home > Add-Ons > Get Add-Ons**, search and install:
- Deep Learning Toolbox
- Deep Learning Toolbox Model for ResNet-18 Network

## 2. Add training images
Drop images into the matching subfolder — filenames don't matter, folder name = class label:
```
vehicle_images/
  sedan/        (put sedan photos here)
  suv/
  motorcycle/
  truck/
  van/
```
Aim for at least 30–50 images per class for a convincing demo (more = better accuracy).
Quick sources: Kaggle "Vehicle Type Recognition" / "Stanford Cars" datasets, or your own
traffic-camera / parking-log photos.

## 3. Train the model
Open `train_vehicle_resnet.m` in MATLAB and click **Run** (or run in Command Window):
```matlab
train_vehicle_resnet
```
This will:
- Split your images 80/20 train/validation
- Fine-tune ResNet-18's last layers for your 5 classes
- Show a live training-progress plot and a confusion matrix
- Save `vehicleResNetModel.mat` in the same folder

## 4. Launch the app
```matlab
VehicleClassifierApp
```
- Click **Upload Image**, pick a photo of a vehicle
- Click **Classify Vehicle**
- See the predicted type, confidence %, and a bar chart of all class probabilities

## Notes for your submission (steps 2–3 of the assignment)
- Take a screenshot of the training-progress plot and confusion matrix (output of step 3) for your PPT
- Take a screenshot of the app window with an uploaded image + prediction (output of step 4)
- In the comment field, note your chosen network: **ResNet (ResNet-18 transfer learning)**
