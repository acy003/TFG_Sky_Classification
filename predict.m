%% Load Dataset
dataset = imageDatastore("predictSet\**\cam.png");
files = dataset.Files;
numFiles = numel(files);

%%
% Choose whether to use panchromatic channel or RGB channel
panchromatic = 1;

% Resize the dataset and save resized images in a new directory
resizedDataset = resizeDataset(files, panchromatic, numFiles);

%%
% Get the size of resized images
inputSize = size(imread(resizedDataset.Files{1}));
numClasses = 15;

% Transform images into a 1 dimensional vector of pixel values
images = transformImagesTo1D(inputSize, resizedDataset);

%%
% Load the network
if panchromatic
    load("SkynetworkY.mat", 'skynetwork');
else
    load("Skynetwork.mat", 'skynetwork');
end

%%
% Predict the classes of the dataset using the network
predictions = predClasses(skynetwork,images);

% Save the predictions in a .csv file
writematrix( predictions, "predictions.csv");

% Performs predictions on the specified dataset using the passed network
function predictions = predClasses(net, dataset)

    %Perform prediction on test set
    y = net(dataset);
    predictions = vec2ind(y);
    
end

% Resizes the given files and saves them in a new directory
% Transforms images into panchromatic images if specified
function resizedDataset = resizeDataset(files, panchromatic,numFiles) 

    if panchromatic
    directory = "resizedPredictY";
    else
    directory = "resizedPredict";
    end

    resizedDataset = 0;
    fileCount = 0;
    %Check if resized folder already exists
    if exist(directory, 'dir')
        fileCount = sum(~[dir(directory).isdir]);
        %Check if all the files are already in the folder
        if fileCount == numFiles
            resizedDataset = imageDatastore(directory);
        end
    end
    % Resize the images if the files are not in the folder or the folder doesnt
    % exist
    if ~exist(directory, 'dir') || (fileCount ~= numFiles)
        mkdir(directory);
    
    
        % Loop through each file and resize the image
        for i = 1:numel(files)
            img = imread(files{i});
            resizedImg = imresize(img, [128 117]);
            [~, filename, ext] = fileparts(files{i});
            resizedFilename = sprintf('%s_resized_%03d%s', filename, i, ext);
            newFilename = fullfile(directory, resizedFilename); % Assuming current directory for storing resized images
            if panchromatic
                resizedImg = rgb2gray(resizedImg);
            end
            imwrite(resizedImg, newFilename);
        end
        resizedDataset = imageDatastore(directory);
    end

end

%Flattens the Image Matrix of pixel values into a 1 dimensional vector
function transformedImages = transformImagesTo1D(inputSize, dataset)

    transformedImages = zeros(prod(inputSize), numel(dataset.Files));
    for i = 1:numel(dataset.Files)
        img = imread(dataset.Files{i});
        transformedImages(:, i) = img(:); % Flatten and store each image in a column
    end

end