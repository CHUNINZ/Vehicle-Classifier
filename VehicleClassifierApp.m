function VehicleClassifierApp
% VEHICLECLASSIFIERAPP  Vehicle Type Classifier (ResNet-based) - MATLAB App
% ------------------------------------------------------------------------
% Upload a traffic-camera or parking-log image and this app tells you
% whether the vehicle is a sedan, SUV, motorcycle, truck, or van.
%
% Run train_vehicle_resnet.m FIRST to produce "vehicleResNetModel.mat".
% Then just run this file (type VehicleClassifierApp in the Command Window,
% or press Run in the Editor).
% ------------------------------------------------------------------------

    %% Load trained model
    modelFile = fullfile(pwd, 'vehicleResNetModel.mat');
    if ~isfile(modelFile)
        error(['vehicleResNetModel.mat not found. Run train_vehicle_resnet.m first ', ...
               'to train and save the model.']);
    end
    modelData = load(modelFile, 'trainedNet', 'classNames', 'inputSize');
    net        = modelData.trainedNet;
    classNames = modelData.classNames;
    inputSize  = modelData.inputSize;

    %% Build UI
    fig = uifigure('Name', 'Vehicle Type Classifier (ResNet)', ...
                    'Position', [100 100 780 560], 'Color', [0.97 0.98 1]);

    uilabel(fig, 'Text', 'Vehicle Type Classifier', ...
        'FontSize', 22, 'FontWeight', 'bold', ...
        'Position', [30 500 500 40]);

    uilabel(fig, 'Text', 'Upload a traffic-camera / parking-log image to classify the vehicle.', ...
        'FontSize', 12, 'FontColor', [0.35 0.35 0.35], ...
        'Position', [30 475 600 25]);

    % Image display axes
    ax = uiaxes(fig, 'Position', [30 150 420 310]);
    title(ax, 'Uploaded Image');
    ax.XTick = []; ax.YTick = [];
    imshow(zeros(224,224,3,'uint8'), 'Parent', ax); % placeholder

    % Upload button
    btnUpload = uibutton(fig, 'Text', 'Upload Image', ...
        'FontSize', 14, 'Position', [30 100 130 40], ...
        'ButtonPushedFcn', @(btn,event) onUpload());

    % Classify button
    btnClassify = uibutton(fig, 'Text', 'Classify Vehicle', ...
        'FontSize', 14, 'Position', [175 100 150 40], ...
        'BackgroundColor', [0.13 0.45 0.85], 'FontColor', 'white', ...
        'Enable', 'off', ...
        'ButtonPushedFcn', @(btn,event) onClassify());

    % Result label
    lblResult = uilabel(fig, 'Text', 'Prediction: -', ...
        'FontSize', 20, 'FontWeight', 'bold', 'FontColor', [0.1 0.5 0.2], ...
        'Position', [470 470 280 35]);

    lblConfidence = uilabel(fig, 'Text', 'Confidence: -', ...
        'FontSize', 14, 'Position', [470 440 280 25]);

    % Bar chart of class probabilities
    axBar = uiaxes(fig, 'Position', [470 150 280 270]);
    title(axBar, 'Class Probabilities');
    axBar.XTickLabelRotation = 30;

    % App state
    currentImage = [];

    %% Callbacks
    function onUpload()
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp', 'Image Files'}, ...
            'Select a vehicle image');
        if isequal(file, 0)
            return; % user cancelled
        end
        fullPath = fullfile(path, file);
        img = imread(fullPath);
        currentImage = img;

        imshow(img, 'Parent', ax);
        title(ax, 'Uploaded Image');

        lblResult.Text = 'Prediction: -';
        lblConfidence.Text = 'Confidence: -';
        cla(axBar);

        btnClassify.Enable = 'on';
    end

    function onClassify()
        if isempty(currentImage)
            uialert(fig, 'Please upload an image first.', 'No Image');
            return;
        end

        % Preprocess: resize + ensure RGB, matching ResNet input
        imgResized = imresize(currentImage, inputSize(1:2));
        if size(imgResized, 3) == 1
            imgResized = repmat(imgResized, 1, 1, 3);
        end

        [label, scores] = classify(net, imgResized);
        [sortedScores, idx] = sort(scores, 'descend');
        sortedNames = classNames(idx);

        lblResult.Text = ['Prediction: ' char(label)];
        lblConfidence.Text = sprintf('Confidence: %.1f%%', sortedScores(1) * 100);

        bar(axBar, categorical(sortedNames, sortedNames), sortedScores);
        ylim(axBar, [0 1]);
        ylabel(axBar, 'Probability');
        title(axBar, 'Class Probabilities');
    end

end
