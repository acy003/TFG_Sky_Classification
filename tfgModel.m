%%Load Dataset
dataset = imageDatastore("Dataset\**\cam.png", LabelSource="foldernames");

%Set Labels
files = dataset.Files;
dataset.Labels = setLabels(files);



%%
%Variable to decide whether images should be processed into panchromatic
%channel after resizing or not
panchromatic = false;
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
        fileCount = sum(~[dir("resizedImages").isdir]);
        %Check if all the files are already in the folder
        if fileCount == 1500
            resizedDataset = imageDatastore("resizedImages\");
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

function net = setupNetwork(layers, maxVal, epochs, trainfcn)
    
    net = patternnet(layers,trainfcn);
    net.trainParam.max_fail = maxVal;
    net.trainParam.epochs = epochs;
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;

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
        headers = {"DateTime","Accuracy",  "Hidden Layers", "Hidden Neurons",  "Training Function"};
        fprintf(fid, "%s, %s, %s,%s,%s \n", headers{:});
    else
        % If the file exists, open it to append data
        fid = fopen(filename, 'a');
    end
    
    % Write the data to the file
    fprintf(fid, '%s,%4f,%d,%d,%s\n', results{1}, results{2},results{3},results{4},results{5});
    
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
                net = setupNetwork(layers,maxVal,epochs,trainfcn);
                       
                %Train Neural Network with specified training function
                rng(155);
                [net, tr] = train(net,trainImages, trainLabels, 'useParallel', 'yes');
                
                % Extract testing set using training record
                testImages = trainImages(:, tr.testInd);
                testLabels = trainLabels(:, tr.testInd);
    
                %Perform prediction on test set
                y = net(testImages);
                predictedClasses = vec2ind(y);
                
                %Calculate accuracy
                actualClasses = vec2ind(testLabels);
                accuracy=sum(predictedClasses==actualClasses)/size(testImages,2)*100;
                
                %Get confusion matrices using predicted and true values
                cmatrix = confusionmat(actualClasses, predictedClasses);
                
                
                % Define the data to be written, including the current date and time
                currentDateTime = datetime('now');
                results = {string(currentDateTime), accuracy, hiddenLayers, hiddenNeurons, funcName};

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