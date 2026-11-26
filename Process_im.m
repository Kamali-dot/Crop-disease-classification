function [result,feat_disease] = Process_im(I, Allfeat_disease, classLabels)

%======================================================================
% MAIN PROCESSING
%======================================================================

[result,feat_disease] = leaf_disease_process( ...
    I,Allfeat_disease,classLabels);

end

function [result,features] = leaf_disease_process(I,training,classLabels)

H = 256;
W = 256;

%--------------------------------------------------------------
% 1. CONTRAST ENHANCEMENT
%--------------------------------------------------------------

Ienh = contrast_stretch(I);

%--------------------------------------------------------------
% 2. RGB TO LAB
%--------------------------------------------------------------

[A,B] = rgb_to_lab_ab(Ienh);

%--------------------------------------------------------------
% 3. K-MEANS CLUSTERING
%--------------------------------------------------------------

[labels,centers] = kmeans_three(A,B,5);

%--------------------------------------------------------------
% 4. SELECT CLUSTER WITH MAXIMUM a*
%--------------------------------------------------------------

selected = 1;

if centers(2,1) > centers(selected,1)
    selected = 2;
end

if centers(3,1) > centers(selected,1)
    selected = 3;
end

%--------------------------------------------------------------
% 5. SEGMENT IMAGE
%--------------------------------------------------------------

seg = zeros(H,W,3,'uint8');

for r = 1:H

    for c = 1:W

        if labels(r,c) == selected

            seg(r,c,1) = Ienh(r,c,1);
            seg(r,c,2) = Ienh(r,c,2);
            seg(r,c,3) = Ienh(r,c,3);

        end

    end

end

%--------------------------------------------------------------
% 6. RGB TO GRAYSCALE
%--------------------------------------------------------------

gray = rgb_gray(seg);

%--------------------------------------------------------------
% 7. AREA CALCULATION
%--------------------------------------------------------------

diseaseArea = 0;
leafArea = 0;

for r = 1:H

    for c = 1:W

        if labels(r,c) == selected
            diseaseArea = diseaseArea + 1;
        end

        if Ienh(r,c,1) ~= 0 || ...
           Ienh(r,c,2) ~= 0 || ...
           Ienh(r,c,3) ~= 0

            leafArea = leafArea + 1;

        end

    end

end

if leafArea == 0

    affected = 0;

else

    affected = diseaseArea/leafArea;

end

% Same correction as original code
if affected < 0.1
    affected = affected + 0.15;
end

%--------------------------------------------------------------
% 8. GLCM
%--------------------------------------------------------------

G = make_glcm(gray);

%--------------------------------------------------------------
% 9. GLCM FEATURES
%--------------------------------------------------------------

[contrast,correlation,energy,homogeneity] = ...
    calculate_glcm_features(G);

%--------------------------------------------------------------
% 10. STATISTICAL FEATURES
%--------------------------------------------------------------

[meanValue,stdValue,entropyValue,rmsValue,...
    varianceValue,smoothness,kurtosisValue,...
    skewnessValue] = ...
    statistical_features(gray);

%--------------------------------------------------------------
% 11. IDM
%--------------------------------------------------------------

IDM = calculate_idm(gray);

%--------------------------------------------------------------
% 12. 14 FEATURE VECTOR
%--------------------------------------------------------------

features = zeros(1,14);

features(1)  = affected*100;
features(2)  = contrast;
features(3)  = correlation;
features(4)  = energy;
features(5)  = homogeneity;
features(6)  = meanValue;
features(7)  = stdValue;
features(8)  = entropyValue;
features(9)  = rmsValue;
features(10) = varianceValue;
features(11) = smoothness;
features(12) = kurtosisValue;
features(13) = skewnessValue;
features(14) = IDM;

%--------------------------------------------------------------
% 13. KNN
%--------------------------------------------------------------

result = custom_knn(features,training,classLabels,5);

end


%======================================================================
% CUSTOM IMAGE RESIZE
%
% Nearest-neighbour interpolation
%======================================================================

function J = resize_nearest(I,newH,newW)

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

end


%======================================================================
% CONTRAST STRETCH
%
% y = (x-min)/(max-min) * 255
%======================================================================

function J = contrast_stretch(I)

H = size(I,1);
W = size(I,2);

J = zeros(H,W,3,'uint8');

for ch = 1:3

    minimum = uint8(255);
    maximum = uint8(0);

    for r = 1:H

        for c = 1:W

            value = (I(r,c,ch));

            if value < minimum
                minimum = value;
            end

            if value > maximum
                maximum = value;
            end

        end

    end

    if maximum == minimum

        for r = 1:H
            for c = 1:W
                J(r,c,ch) = I(r,c,ch);
            end
        end

    else

        for r = 1:H

            for c = 1:W

                value = (I(r,c,ch));

                output = ...
                    ((value-minimum)/ ...
                    (maximum-minimum))*uint8(255);

                if output < uint8(0)
                    output = uint8(0);
                end

                if output > uint8(255)
                    output = uint8(255);
                end

                J(r,c,ch) = uint8(output);

            end

        end

    end

end

end


%======================================================================
% RGB TO LAB a* AND b*
%======================================================================

function [A,B] = rgb_to_lab_ab(I)

H = size(I,1);
W = size(I,2);

A = zeros(H,W);
B = zeros(H,W);

Xn = 0.95047;
Yn = 1.00000;
Zn = 1.08883;

for r = 1:H

    for c = 1:W

        R =  (I(r,c,1))/255;
        G =  (I(r,c,2))/255;
        Bl =  (I(r,c,3))/255;

        R = gamma_inverse(R);
        G = gamma_inverse(G);
        Bl = gamma_inverse(Bl);

        X = ...
            0.4124564*R + ...
            0.3575761*G + ...
            0.1804375*Bl;

        Y = ...
            0.2126729*R + ...
            0.7151522*G + ...
            0.0721750*Bl;

        Z = ...
            0.0193339*R + ...
            0.1191920*G + ...
            0.9503041*Bl;

        fx = lab_function(X/Xn);
        fy = lab_function(Y/Yn);
        fz = lab_function(Z/Zn);

        A(r,c) = 500*(fx-fy);

        B(r,c) = 200*(fy-fz);

    end

end

end


%======================================================================
% GAMMA INVERSE
%======================================================================

function y = gamma_inverse(x)

if x <= 0.04045

    y = x/12.92;

else

    y = x*x;

    y = y/(1.055*1.055);

end

end


%======================================================================
% LAB FUNCTION
%======================================================================

function y = lab_function(x)

delta = 6/29;

if x > delta^3

    y = cube_root_custom(x);

else

    y = x/(3*delta^2)+4/29;

end

end


%======================================================================
% CUBE ROOT
%
% Newton-Raphson method
%======================================================================

function y = cube_root_custom(x)

if x <= uint8(0)

    y = uint8(0);
    return;

end

y = uint8(1);

for k = 1:8

    y = (2*y+x/(y*y))/3;

end

end


%======================================================================
% THREE CLUSTER K-MEANS
%======================================================================

function [labels,centers] = ...
    kmeans_three(A,B,iterations)

H = size(A,1);
W = size(A,2);

labels = zeros(H,W);

centers = zeros(3,2);

%--------------------------------------------------------------
% Find minimum and maximum a*
%--------------------------------------------------------------

amin = A(1,1);
amax = A(1,1);

for r = 1:H

    for c = 1:W

        if A(r,c) < amin
            amin = A(r,c);
        end

        if A(r,c) > amax
            amax = A(r,c);
        end

    end

end

%--------------------------------------------------------------
% Initial centers
%--------------------------------------------------------------

centers(1,1) = amin;
centers(1,2) = 0;

centers(2,1) = (amin+amax)/2;
centers(2,2) = 0;

centers(3,1) = amax;
centers(3,2) = 0;

%--------------------------------------------------------------
% K-MEANS ITERATIONS
%--------------------------------------------------------------

for iteration = 1:iterations

    sumA = zeros(3,1);
    sumB = zeros(3,1);
    count = zeros(3,1);

    %----------------------------------------------------------
    % Assignment
    %----------------------------------------------------------

    for r = 1:H

        for c = 1:W

            d1 = ...
                (A(r,c)-centers(1,1))^2 + ...
                (B(r,c)-centers(1,2))^2;

            d2 = ...
                (A(r,c)-centers(2,1))^2 + ...
                (B(r,c)-centers(2,2))^2;

            d3 = ...
                (A(r,c)-centers(3,1))^2 + ...
                (B(r,c)-centers(3,2))^2;

            if d1 <= d2 && d1 <= d3

                k = 1;

            elseif d2 <= d1 && d2 <= d3

                k = 2;

            else

                k = 3;

            end

            labels(r,c) = k;

            sumA(k) = sumA(k)+A(r,c);
            sumB(k) = sumB(k)+B(r,c);

            count(k) = count(k)+1;

        end

    end

    %----------------------------------------------------------
    % Center update
    %----------------------------------------------------------

    for k = 1:3

        if count(k) > 0

            centers(k,1) = sumA(k)/count(k);
            centers(k,2) = sumB(k)/count(k);

        end

    end

end

end


%======================================================================
% RGB TO GRAYSCALE
%
% Gray = 0.299R + 0.587G + 0.114B
%======================================================================

function G = rgb_gray(I)

H = size(I,1);
W = size(I,2);

G = zeros(H,W);

for r = 1:H

    for c = 1:W

        R =  (I(r,c,1));
        Gc =  (I(r,c,2));
        B =  (I(r,c,3));

        G(r,c) = ...
            0.299*R + ...
            0.587*Gc + ...
            0.114*B;

    end

end

end


%======================================================================
% GLCM
%
% 8 gray levels
%
% Horizontal neighbour:
% (r,c) and (r,c+1)
%======================================================================

function G = make_glcm(gray)

H = size(gray,1);
W = size(gray,2);

G = zeros(8,8);

for r = 1:H

    for c = 1:W-1

        p = gray_level(gray(r,c));
        q = gray_level(gray(r,c+1));

        G(p,q) = G(p,q)+1;

    end

end

end


%======================================================================
% GRAY LEVEL QUANTIZATION
%======================================================================

function level = gray_level(x)

if x < 32

    level = 1;

elseif x < 64

    level = 2;

elseif x < 96

    level = 3;

elseif x < 128

    level = 4;

elseif x < 160

    level = 5;

elseif x < 192

    level = 6;

elseif x < 224

    level = 7;

else

    level = 8;

end

end


%======================================================================
% GLCM FEATURES
%
% Contrast:
% SUM((i-j)^2 P(i,j))
%
% Energy:
% SUM(P(i,j)^2)
%
% Homogeneity:
% SUM(P(i,j)/(1+abs(i-j)))
%
% Correlation:
% numerator/(sigma_i*sigma_j)
%======================================================================

function [contrast,correlation,energy,homogeneity] = ...
    calculate_glcm_features(G)

total = 0;

for i = 1:8

    for j = 1:8

        total = total+G(i,j);

    end

end

if total == 0

    contrast = 0;
    correlation = 0;
    energy = 0;
    homogeneity = 0;

    return;

end

%--------------------------------------------------------------
% Means
%--------------------------------------------------------------

meanI = 0;
meanJ = 0;

for i = 1:8

    for j = 1:8

        p = G(i,j)/total;

        meanI = meanI+(i-1)*p;
        meanJ = meanJ+(j-1)*p;

    end

end

%--------------------------------------------------------------
% Features
%--------------------------------------------------------------

contrast = 0;
energy = 0;
homogeneity = 0;

varianceI = 0;
varianceJ = 0;

numerator = 0;

for i = 1:8

    for j = 1:8

        p = G(i,j)/total;

        di = (i-1)-meanI;
        dj = (j-1)-meanJ;

        difference = i-j;

        contrast = ...
            contrast+difference^2*p;

        energy = ...
            energy+p^2;

        homogeneity = ...
            homogeneity+p/(1+abs(difference));

        varianceI = ...
            varianceI+di^2*p;

        varianceJ = ...
            varianceJ+dj^2*p;

        numerator = ...
            numerator+di*dj*p;

    end

end

stdI = sqrt_custom(varianceI);
stdJ = sqrt_custom(varianceJ);

if stdI > 0 && stdJ > 0

    correlation = ...
        numerator/(stdI*stdJ);

else

    correlation = 0;

end

end


%======================================================================
% STATISTICAL FEATURES
%======================================================================

function [meanValue,stdValue,entropyValue,rmsValue,...
    varianceValue,smoothness,kurtosisValue,skewnessValue] = ...
    statistical_features(gray)

H = size(gray,1);
W = size(gray,2);

N = H*W;

sumX = 0;
sumX2 = 0;

histogram = zeros(8,1);

%--------------------------------------------------------------
% FIRST PASS
%--------------------------------------------------------------

for r = 1:H

    for c = 1:W

        x = gray(r,c);

        sumX = sumX+x;

        sumX2 = sumX2+x*x;

        level = gray_level(x);

        histogram(level) = ...
            histogram(level)+1;

    end

end

%--------------------------------------------------------------
% MEAN
%--------------------------------------------------------------

meanValue = sumX/N;

%--------------------------------------------------------------
% VARIANCE
%
% Var = E[X^2] - E[X]^2
%--------------------------------------------------------------

varianceValue = ...
    sumX2/N-meanValue^2;

if varianceValue < 0
    varianceValue = 0;
end

%--------------------------------------------------------------
% STANDARD DEVIATION
%--------------------------------------------------------------

stdValue = ...
    sqrt_custom(varianceValue);

%--------------------------------------------------------------
% RMS
%
% RMS = sqrt(E[X^2])
%--------------------------------------------------------------

rmsValue = ...
    sqrt_custom(sumX2/N);

%--------------------------------------------------------------
% ENTROPY
%
% H = -SUM(p*log2(p))
%--------------------------------------------------------------

entropyValue = uint32(0);

for k = 1:8

    if histogram(k) > 0

        p = histogram(k)/N;

        entropyValue = ...
            entropyValue-p*log2_custom(p);

    end

end

%--------------------------------------------------------------
% SMOOTHNESS
%
% Original project equation:
%
% Smoothness = 1 - 1/(1+a)
%--------------------------------------------------------------

a = sumX;

smoothness = ...
    1-1/(1+a);

%--------------------------------------------------------------
% THIRD AND FOURTH CENTRAL MOMENTS
%--------------------------------------------------------------

mu3 = 0;
mu4 = 0;

for r = 1:H

    for c = 1:W

        difference = ...
            gray(r,c)-meanValue;

        d2 = difference^2;

        mu3 = ...
            mu3+d2*difference;

        mu4 = ...
            mu4+d2*d2;

    end

end

mu3 = mu3/N;
mu4 = mu4/N;

%--------------------------------------------------------------
% SKEWNESS
%--------------------------------------------------------------

if stdValue > 0

    skewnessValue = ...
        mu3/(stdValue^3);

else

    skewnessValue = 0;

end

%--------------------------------------------------------------
% KURTOSIS
%--------------------------------------------------------------

if varianceValue > 0

    kurtosisValue = ...
        mu4/(varianceValue^2);

else

    kurtosisValue = 0;

end

end


%======================================================================
% LOG2
%
% log2(x) = ln(x)/ln(2)
%
% Natural logarithm calculated using series expansion.
%======================================================================

function y = log2_custom(x)
% HDL-compatible integer log2
% No while
% No return

y = uint32(0);

for k = 0:31

    if uint32(x) >= bitshift(uint32(1),k)

        y = uint32(k);

    end

end

end


%======================================================================
% SQUARE ROOT
%
% Newton-Raphson:
%
% y(n+1) = 0.5*(y+x/y)
%======================================================================

function y = sqrt_custom(x)

if x <= 0

    y = 0;

    return;

end

if x >= 1

    y = x;

else

    y = 1;

end

for k = 1:10

    y = 0.5*(y+x/y);

end

end


%======================================================================
% INVERSE DIFFERENCE MOMENT
%
% IDM = SUM[
%
%       image(i,j)
%       -----------------
%       1+(i-j)^2
%
% ]
%======================================================================

function IDM = calculate_idm(gray)

H = size(gray,1);
W = size(gray,2);

IDM = 0;

for i = 1:H

    for j = 1:W

        difference = i-j;

        denominator = ...
            1+difference^2;

        IDM = ...
            IDM+gray(i,j)/denominator;

    end

end

end


%======================================================================
% CUSTOM KNN
%
% K = 5
%
% Distance:
%
% d = SUM((test-feature - train-feature)^2)
%
% Square root is not needed because only distance ordering
% is required.
%======================================================================

function result = ...
    custom_knn(test,training,classLabels,K)

N = size(training,1);
F = size(training,2);

% Make sure only 14 features are used
if F > 14
    F = 14;
end

bestDistance = ...
    ones(K,1)*1.0e30;

bestClass = ...
    ones(K,1)*(-1);

%======================================================================
% CALCULATE DISTANCE TO EVERY TRAINING SAMPLE
%======================================================================

for n = 1:N

    distance = 0;

    for f = 1:F

        difference = ...
            test(f)-training(n,f);

        distance = ...
            distance+difference^2;

    end

    %--------------------------------------------------------------
    % Find insertion position
    %--------------------------------------------------------------

    position = K;

    for k = 1:K

        if distance < bestDistance(k)

            position = k;

            

        end

    end

    %--------------------------------------------------------------
    % Insert distance into sorted list
    %--------------------------------------------------------------

    if position <= K
   
        for k = K:-1:2

            if k >= position + 1

                bestDistance(k) = ...
                    bestDistance(k-1);

                bestClass(k) = ...
                    bestClass(k-1);

            end

        end

        bestDistance(position) = distance;

        bestClass(position) = ...
            uint8(classLabels(n));

    end
  
end

%======================================================================
% MAJORITY VOTING
%======================================================================

votes = zeros(1,5);

for k = 1:K

    currentClass = bestClass(k);

    if currentClass >= 0 && currentClass <= 4

        votes(currentClass+1) = ...
            votes(currentClass+1)+1;

    end

end

%======================================================================
% SELECT MAXIMUM VOTE
%======================================================================

maximumVotes = votes(1);

result = 0;

for k = 2:5

    if votes(k) > maximumVotes

        maximumVotes = votes(k);

        result = k-1;

    end

end

end
