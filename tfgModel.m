%%Load Dataset
dataset = imageDatastore("Dataset\**\cam.png", LabelSource="foldernames");

%Set Labels
files = dataset.Files;
labels = setLabels(files);
dataset.Labels = labels;



%%
%Variable to decide whether images should be processed into panchromatic
%channel after resizing or not
panchromatic = true;
numFiles = numel(files);

%Resize the dataset and save resized images in a new directory
resizedDataset = resizeDataset(files, panchromatic, numFiles);

%%
% Update the Files property of the datastore with the resized filenames
resizedDataset.Labels = labels;
inputSize = size(imread(resizedDataset.Files{1}));

%%
%Set rng seed
myStream = RandStream('mt19937ar','NormalTransform','Polar');
RandStream.setGlobalStream(myStream);
rng(155);

%%
%Transform image into 1D array of pixel values 
trainImages = transformImagesTo1D(inputSize, resizedDataset);

%%
%Change labels to binary values indicating the images class with a 1 in the
%corresponding row indicating the sky type
numClasses = 15;
trainLabels = transformLabels(resizedDataset.Labels, numClasses);


%%
%Set array of neuron quantity to train
neurons = [1 3 5 10 15 20 30 40 50 60 70 80 90 100];
funcs = ["trainscg","trainrp" "trainoss" "traingdx"];
funcNames = ["SCG", "RP", "OSS", "GDX"];

%%
trainCloudNetwork(neurons, funcs, funcNames, trainImages, trainLabels, panchromatic);

%%
% Update the Files property of the datastore with the resized filenames
resizedDataset = resizeDataset(files, 0, numFiles);
resizedDataset.Labels = labels;
inputSize = size(imread(resizedDataset.Files{1}));

%Transform image into 1D array of pixel values 
trainImages = transformImagesTo1D(inputSize, resizedDataset);
%Change labels to binary values indicating the images class with a 1 in the
%corresponding row indicating the sky type
numClasses = 15;
trainLabels = transformLabels(resizedDataset.Labels, numClasses);

trainCloudNetwork(neurons, funcs, funcNames, trainImages, trainLabels, 0);
%%

function [train_idx,val_idx,test_idx] = prfmCV(trainImages, numFolds, fold)
    % Define the number of folds for cross-validation
    num_folds = numFolds;
    
    % Perform cross-validation
    rng(155);
    cv = cvpartition(size(trainImages, 2), 'KFold', num_folds);
    
    % Create a separate test set and exclude its indices from cross-validation
    test_set_size = 0.15; % 15% of data for testing
    rng(155);
    cv_test = cvpartition(size(trainImages, 2), 'HoldOut', test_set_size);
    train_cv_idx = training(cv,fold);
    test_idx = test(cv_test);
    % Remaining samples after excluding the test set for cross-validation
    train_idx = find(train_cv_idx & ~test_idx);
    % Split the remaining samples for cross-validation into training and validation sets
    train_val_idx = test(cv, fold); % Example fold index
    val_idx = find(train_val_idx & ~test_idx);
    test_idx = find(test_idx); % Test set indices

end

function labels = setLabels(files)

    labels = split(extractAfter( files, "Dataset\"), filesep);
    labels = split(labels(:,1));
    labels = split(labels, "tipo");
    labels = split(labels(:,2));
    labels = str2double(labels);

end

function resizedDataset = resizeDataset(files, panchromatic,numFiles) 

    if panchromatic
    directory = "resizedImagesY";
    else
    directory = "resizedImages";
    end

    resizedDataset = 0;
    fileCount = 0;
    %Check if resized folder already exists
    if exist(directory, 'dir')
        fileCount = sum(~[dir(directory).isdir]);
        %Check if all the files are already in the folder
        if fileCount == 1500
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

function transformedImages = transformImagesTo1D(inputSize, dataset)

    transformedImages = zeros(prod(inputSize), numel(dataset.Files));
    for i = 1:numel(dataset.Files)
        img = imread(dataset.Files{i});
        transformedImages(:, i) = img(:); % Flatten and store each image in a column
    end

end

function transformedLabels = transformLabels(labels, numClasses)

    rows = numClasses;
    cols = numel(labels);
    
    % Create an array of zeros
    array = zeros(rows, cols);
    
    % Set the values based on the label numbers
    for i = 1:cols
        array(:, i) = (1:rows == labels(i));
    end
    transformedLabels = array;

end

function net = setupNetwork(layers, maxVal, epochs, trainfcn, train_idx, val_idx, test_idx)
    
    net = patternnet(layers,trainfcn);
    net.divideFcn = 'divideind';
    net.trainParam.max_fail = maxVal;
    net.trainParam.epochs = epochs;
    net.divideParam.trainInd = train_idx;
    net.divideParam.valInd = val_idx;
    net.divideParam.testInd = test_idx;

end

function saveResults(results, panchromatic)
    
    % Specify the file name
    if panchromatic
        filename = 'resultsY.csv';
    else
        filename = 'results.csv';
    end 

    % Check if the file exists
    if exist(filename, 'file') == 0
        % If the file doesn't exist, create a new one and write the headers
        fid = fopen(filename, 'w');
        headers = {"DateTime","Accuracy","Precision","Recall","Hidden Layers", "Hidden Neurons", "Training Function","Training Time"};
        fprintf(fid, "%s, %s,%s,%s,%s, %s,%s,%s \n", headers{:});
    else
        % If the file exists, open it to append data
        fid = fopen(filename, 'a');
    end
    
    % Write the data to the file
    fprintf(fid, '%s,%4f,%4f,%4f,%d,%d,%s,%4f\n', results{:});
    
    % Close the file
    fclose(fid);
    % Close all open file handles
    fclose('all');
    
    clear headers results fid filename filename

end

function saveCMatrix(cmatrix,panchromatic,hiddenNeurons, hiddenLayers, funcName)

    if panchromatic
        %Create new Folder for confusion matrix
        outputFolder = 'confusionMatricesY';
    else
        %Create new Folder for confusion matrix
        outputFolder = 'confusionMatrices';
    end
    if ~exist(outputFolder, 'dir')
        mkdir(outputFolder);
    end
    
    
    % Specify the file name for the image
    currentDate = string(datetime('now','Format','yyyy-MM-dd_HH_mm_ss'));
    
    filename = sprintf('%s_cMatrix_%s_HN%d_HL%d.png',funcName, string(currentDate), hiddenNeurons, hiddenLayers);
    imageFilename = fullfile(outputFolder, filename);

    % Save the confusion matrix plot as an image
    cChart = confusionchart(cmatrix);
    saveas(cChart, imageFilename);

    clear cChart filename imageFilename currentDate outputFolder

end

function trainCloudNetwork(neurons, functions, funcNames, trainImages, trainLabels, panchromatic)

    for f = 1:length(functions)

        accuraciesTotal = zeros(length(neurons));
        networks = cell(length(neurons),1);
        
        for n = 1:length(neurons)
                %Training options
                hiddenLayers= 1;
                hiddenNeurons = neurons(n);
                layers = repelem(hiddenNeurons, hiddenLayers);
                
                trainfcn = functions(f);
                funcName = funcNames(f);
                
                %Create Network
                maxVal = 10;
                epochs = 1000;
                numFolds = 5;
                accuracies = zeros(numFolds,1);
                precisions = zeros(numFolds,1);
                recalls = zeros(numFolds,1);
                cmatrices = zeros(15, 15);
                avgTime = zeros(numFolds,1);

                for fold = 1:numFolds
                    [train_idx,val_idx,test_idx] = prfmCV(trainImages, numFolds,fold);
                    net = setupNetwork(layers,maxVal,epochs,trainfcn,train_idx,val_idx,test_idx);
                           
                    %Train Neural Network with specified training function
                    rng(155);
                    tic;
                    [net, tr] = train(net,trainImages, trainLabels, 'useParallel', 'yes');
                    avgTime(fold) = toc;
                    % Extract testing set using training record
                    testImages = trainImages(:, tr.testInd);
                    testLabels = trainLabels(:, tr.testInd);
        
                    %Perform prediction on test set
                    y = net(testImages);
                    predictedClasses = vec2ind(y);
                    
                    
                    %Calculate accuracy
                    actualClasses = vec2ind(testLabels);
                    accuracies(fold) =sum(predictedClasses==actualClasses)/size(testImages,2)*100;
                    
                    %Get confusion matrices using predicted and true values
                    cmatrix = confusionmat(actualClasses, predictedClasses);
                    % Accumulate confusion matrix
                    cmatrices = cmatrices + cmatrix;

                    % Calculate precision and recall for each class
                    precision = zeros(15, 1);
                    recall = zeros(15, 1);
                    for i = 1:15
                        TP = cmatrix(i, i);
                        FP = sum(cmatrix(:, i)) - TP;
                        FN = sum(cmatrix(i, :)) - TP;
                        
                        if (TP + FP) == 0
                            precision(i) = 0;
                        else
                            precision(i) = TP / (TP + FP);
                        end

                        if (TP + FN) == 0
                            recall(i) = 0;
                        else
                            recall(i) = TP / (TP + FN);
                        end

                    end
                    precisions(fold) = mean(precision);
                    recalls(fold) = mean(recall);
                    
                end

                precision = mean(precisions);
                recall = mean(recalls);
                accuracy = mean(accuracies);
                accuraciesTotal(n) = accuracy;
                networks{n} = net;
                time = mean(avgTime);
                cmatrix = round(cmatrices/numFolds);
                
                % Define the data to be written, including the current date and time
                currentDateTime = datetime('now');
                results = {string(currentDateTime), accuracy,precision,recall, hiddenLayers, hiddenNeurons, funcName, time};
                
                % Save the data inside a .csv file
                saveResults(results,panchromatic);
                
                % Save the cMatrix in a separate folder
                saveCMatrix(cmatrix,panchromatic,hiddenNeurons,hiddenLayers, funcName);
                
                %Clear variables
                clear currentDateTime results epochs maxVal 
                clear hiddenNeurons hiddenLayers layers trainfcn net y predictedClasses
                clear actualClasses accuracy cmatrix testImages testLabels tr
            
        end
    end
end