%% train_vehicle_resnet.m
% ------------------------------------------------------------------------
% Vehicle Type Classifier - ResNet Transfer Learning (MATLAB)
% Classes: sedan | suv | motorcycle | truck | van
%
% REQUIREMENTS (Add-Ons > Get Add-Ons, search and install):
%   - Deep Learning Toolbox
%   - Deep Learning Toolbox Model for ResNet-18 Network
%
% FOLDER STRUCTURE EXPECTED (relative to this script):
%   vehicle_images/
%       sedan/        *.jpg, *.png ...
%       suv/
%       motorcycle/
%       truck/
%       van/
%
% Each subfolder name becomes the class label automatically.
% Put at least ~30-50 images per class for a reasonable demo; more is better.
% ------------------------------------------------------------------------

clear; clc; close all;

%% 1. Load images into a labeled datastore
dataFolder = fullfile(pwd, 'vehicle_images');
assert(isfolder(dataFolder), ...
    'Folder "vehicle_images" not found next to this script. Create it and add class subfolders.');

imds = imageDatastore(dataFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

fprintf('Found %d images across %d classes:\n', numel(imds.Files), numel(categories(imds.Labels)));
countEachLabel(imds)

%% 2. Split into training (80%) and validation (20%) sets
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

%% 3. Load pretrained ResNet-18
net = resnet18;
inputSize = net.Layers(1).InputSize;   % typically [224 224 3]
lgraph = layerGraph(net);

classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

%% 4. Replace the final layers for our 5 vehicle classes
% ResNet-18's last learnable layer is 'fc1000', final classifier is 'ClassificationLayer_predictions'
newFC = fullyConnectedLayer(numClasses, ...
    'Name', 'fc_vehicle', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);

newClassLayer = classificationLayer('Name', 'classoutput_vehicle');

lgraph = replaceLayer(lgraph, 'fc1000', newFC);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);

%% 5. Data augmentation + resize to match ResNet input size
augmenter = imageDataAugmenter( ...
    'RandRotation', [-15 15], ...
    'RandXReflection', true, ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'ColorPreprocessing', 'gray2rgb', ...
    'DataAugmentation', augmenter);

augimdsVal = augmentedImageDatastore(inputSize(1:2), imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

%% 6. Training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 12, ...
    'InitialLearnRate', 3e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', 10, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

%% 7. Train (fine-tune) the network
fprintf('Starting ResNet-18 fine-tuning...\n');
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

%% 8. Evaluate on the validation set
[predLabels, scores] = classify(trainedNet, augimdsVal);
trueLabels = imdsVal.Labels;
accuracy = mean(predLabels == trueLabels);
fprintf('Validation accuracy: %.2f%%\n', accuracy * 100);

figure;
confusionchart(trueLabels, predLabels, 'Title', 'Vehicle Type Classifier - Confusion Matrix');

%% 9. Save the trained model + class names for the app
save('vehicleResNetModel.mat', 'trainedNet', 'classNames', 'inputSize');
fprintf('Model saved to vehicleResNetModel.mat\n');
fprintf('Now run VehicleClassifierApp.m to launch the app.\n');
