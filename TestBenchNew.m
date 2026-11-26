clc;
clear;
close all;

I = imread('8.jpg');

newH=256;newW=256;
oldH = size(I,1);
oldW = size(I,2);

J = zeros(newH,newW,3,'uint8');

for r = 1:newH

    srcR = floor((r-1)*oldH/newH)+1;

    if srcR > oldH
        srcR = oldH;
    end

    for c = 1:newW

        srcC = floor((c-1)*oldW/newW)+1;

        if srcC > oldW
            srcC = oldW;
        end

        J(r,c,1) = I(srcR,srcC,1);
        J(r,c,2) = I(srcR,srcC,2);
        J(r,c,3) = I(srcR,srcC,3);

    end

end

disp('Input image loaded.');
disp('Image size = 256 x 256');

%======================================================================
% LOAD TRAINING DATABASE
%======================================================================

load('DatabasenTomato_1.mat');

disp('Training database loaded.');

%======================================================================
% MAIN PROCESSING
%======================================================================

[result,feat_disease] = Process_im( ...
    J,Allfeat_disease,class);

%======================================================================
% DISPLAY FEATURES
%======================================================================

disp(' ');
disp('==========================================================');
disp('                 EXTRACTED FEATURES');
disp('==========================================================');

disp(['Affected Area       = ',num2str(feat_disease(1))]);
disp(['Contrast            = ',num2str(feat_disease(2))]);
disp(['Correlation         = ',num2str(feat_disease(3))]);
disp(['Energy              = ',num2str(feat_disease(4))]);
disp(['Homogeneity         = ',num2str(feat_disease(5))]);
disp(['Mean                = ',num2str(feat_disease(6))]);
disp(['Standard Deviation  = ',num2str(feat_disease(7))]);
disp(['Entropy             = ',num2str(feat_disease(8))]);
disp(['RMS                 = ',num2str(feat_disease(9))]);
disp(['Variance            = ',num2str(feat_disease(10))]);
disp(['Smoothness          = ',num2str(feat_disease(11))]);
disp(['Kurtosis            = ',num2str(feat_disease(12))]);
disp(['Skewness            = ',num2str(feat_disease(13))]);
disp(['IDM                 = ',num2str(feat_disease(14))]);

%======================================================================
% DISPLAY CLASSIFICATION
%======================================================================

disp(' ');
disp('==========================================================');
disp('              CLASSIFICATION RESULT');
disp('==========================================================');

if result == 0

    disp('Class : 0');
    disp('Disease : Alternaria Alternata');

elseif result == 1

    disp('Class : 1');
    disp('Disease : Anthracnose');

elseif result == 2

    disp('Class : 2');
    disp('Disease : Bacterial Blight');

elseif result == 3

    disp('Class : 3');
    disp('Disease : Cercospora Leaf Spot');

elseif result == 4

    disp('Class : 4');
    disp('Disease : Healthy Leaf');

else

    disp('Unknown Class');

end

disp('==========================================================');

% for i=1:17
%     I = imread([num2str(i) '.jpg']);
% I=imresize(I,[256 256]);
% 
%     Allfeat_disease(i,:) = NewDatabase(I);
%     i
% end
