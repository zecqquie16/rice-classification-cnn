close all
%unzip('rice.zip');
ratio = 0.88;
%Final form by using the best parameters/ conditions found%
imds = imageDatastore('Rice_Image_Dataset','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,ratio,'randomized');
minImagesPerClass = floor(1100 / numel(unique(imds.Labels)));

imds = splitEachLabel(imds, minImagesPerClass, 'randomized');

[imdsTrain,imdsValidation] = splitEachLabel(imds,ratio,'randomized');
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
net = alexnet;

analyzeNetwork(net)
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels))

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('adam', ...
    'MiniBatchSize',28, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);

[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label1 = YPred(idx(i));
    label2 = YValidation(idx(i));
    label = ["Actual ->", label1, newline, " Pred -> ",label2];
    title(string(label));
end

YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)
%% Different Architecture approach

close all
%unzip('rice.zip');
ratio = 0.88;
%Use of the same parameters in the original achitecture (optimal parameters)%
imds = imageDatastore('Rice_Image_Dataset','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,ratio,'randomized');
minImagesPerClass = floor(300 / numel(unique(imds.Labels)));

imds = splitEachLabel(imds, minImagesPerClass, 'randomized');

[imdsTrain,imdsValidation] = splitEachLabel(imds,ratio,'randomized');
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
net = alexnet;

analyzeNetwork(net)
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels))
%Here we keep the layers of AlexNet apart from the three last 
%We add a fully connected layer which will take far more neurons in order
%to adapt from the previous layers of AlexNet
%following the new fc layer we add an activation function here Relu And a
%drop out layer.
layers2 = [
    layersTransfer
    fullyConnectedLayer(512, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor', 20, 'BiasLearnRateFactor', 20)
    softmaxLayer
    classificationLayer
];
 pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('adam', ...
    'MiniBatchSize',28, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers2,options);

[YPred,scores] = classify(netTransfer,augimdsValidation);
YValidation = imdsValidation.Labels;
idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label1 = YPred(idx(i));
    label2 = YValidation(idx(i));
    label = ["Actual ->", label1, newline, " Pred -> ",label2];
    title(string(label));
end

YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)
%% close all
%unzip('rice.zip');
ratio = 0.88;
%finding optimal training options
imds = imageDatastore('Rice_Image_Dataset','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain, imdsValidation] = splitEachLabel(imds, ratio, 'randomized');
minImagesPerClass = floor(150 / numel(unique(imds.Labels)));

imds = splitEachLabel(imds, minImagesPerClass, 'randomized');
[imdsTrain, imdsValidation] = splitEachLabel(imds, ratio, 'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain, idx(i));
    imshow(I)
end

net = alexnet;
analyzeNetwork(net)
inputSize = net.Layers(1).InputSize;
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels));

layers = [  
    layersTransfer 
    fullyConnectedLayer(numClasses, 'WeightLearnRateFactor',20, 'BiasLearnRateFactor',20)
    softmaxLayer 
    classificationLayer
];

pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, 'DataAugmentation', imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);

% To simplify we will make lr and bs only vary within a set of few values
BatchSizesValues = [16, 28, 32, 40];
LRvalues = [1e-3,1e-4,1e-5];

bestAccuracy = 0;
bestBS = NaN;
bestLR = NaN;
results = [];  % storage in order to keep track

for bs = BatchSizesValues
    for lr = LRvalues
        options = trainingOptions('adam', ...
            'MiniBatchSize', bs, ...
            'MaxEpochs', 6, ...
            'InitialLearnRate', lr, ...
            'Shuffle', 'every-epoch', ...
            'ValidationData', augimdsValidation, ...
            'ValidationFrequency', 3, ...
            'Verbose', false, ...
            'Plots', 'none'); 
        fprintf('Training with = %d and LR = %.1e...\n', bs, lr);
        % Testing a single condition at a time
        netTemp = trainNetwork(augimdsTrain, layers, options);
        YPredTemp = classify(netTemp, augimdsValidation);
        accuracy = mean(YPredTemp == imdsValidation.Labels);
        fprintf('Accuracy  %.2f%%\n', acc*100);
        % compare each iteration's accuracy to compare and stock best
        % combination.
        results = [results; bs, lr, accuracy];
        if accuracy > bestAccuracy
            bestAccuracy = accuracy;
            bestBS = bs;
            bestLR = lr;
        end
    end
end
%By running this section of code we can conclude that
%best learning rate is 1e-4
%worst learning rate is 1e-3
%batch sizes are balanced however batch size within [28,32] are more
%reliable.
fprintf('\nBest parameters :BatchSize = %d, LR = %.1e, Accuracy = %.2f%%\n', bestBS, bestLR, bestAccuracy*100);





