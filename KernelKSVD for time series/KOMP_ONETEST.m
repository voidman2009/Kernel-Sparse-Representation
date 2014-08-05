close all;
clear all;
clc;

tr_dat = load('E:\graduate learning\UCRdataset\Coffee_TRAIN');
tt_dat = load('E:\graduate learning\UCRdataset\Coffee_TEST');
tr_label = tr_dat(:,1);
tt_label = tt_dat(:,1);
[m1,n1] = size(tr_dat);
tr_dat = tr_dat(1:m1,2:n1)';
[m2,n2] = size(tt_dat);
tt_dat = tt_dat(1:m2,2:n2)';

tr_dat = zscore(tr_dat);
tt_dat = zscore(tt_dat);
%If the labels are not starting at 1, formalize it.
tr_label = tr_label + 1;
% for iter = 1:m1
%    if tr_label(iter) == -1
%        tr_label(iter) = 1;
%    else
%        tr_label(iter) = 2;
%    end
% end
tt_label = tt_label + 1;
% for iter = 1:m2
%    if tt_label(iter) == -1
%        tt_label(iter) = 1;
%    else
%        tt_label(iter) = 2;
%    end
% end

classNum = 2;           %Number of classes
lamda = n1 ;            %Gaussian RBF parameters

Y = tr_dat;
A = eye(size(Y, 2));
dimension = size(Y, 2);
YTY = zeros(dimension, dimension);              %Calculation of K(Y, Y)
for k1 = 1:dimension
    for k2 = 1:dimension
        if k1 == k2
            %YTY(k1, k2) = DTW(Y(:, k1), 0, 0);  %Linear kernel
            YTY(k1, k2) = 1;                     %Gaussian RBF kernel
        else
            if k1 < k2
                %YTY(k1, k2) = 0.5 * (DTW(Y(:, k1), 0, 0) + DTW(Y(:, k2), 0, 0) - DTW(Y(:, k1), Y(:, k2), 1));
                YTY(k1, k2) = exp(-1 * (ERP(Y(:, k1), Y(:, k2), 1) ^ 2) / lamda);
                YTY(k2, k1) = YTY(k1, k2);
            end
        end
    end
end
    
for s = 1:8                     
    sparsity = s;
    residual = zeros(m2, classNum);
    A = eye(size(Y, 2));

    %This procedure is referred in Fig. 1 of the original paper
    for j = 1:m2
        z = tt_dat(:, j);
        zTY = zeros(1, dimension);
        for k = 1:dimension
            %zTY(1, k) = 0.5 * (DTW(z, 0, 0) + DTW(Y(:, k), 0, 0) - DTW(z, Y(:, k), 1));
            zTY(1, k) = exp(-1 * (ERP(z, Y(:, k), 1) ^ 2) / lamda);
        end
        [x, zTY, YTY] = KOMP_ONE(0, zTY, YTY, A, sparsity);
        %Calculation of K(z, z)
        zTz = 1;
        for i = 1:classNum
            code = x(tr_label == i);
            residual(j, i) = zTz - 2 * zTY(tr_label == i) * code + code' *  YTY(tr_label == i, tr_label == i) * code;
        end
    end

    %Classification, finding the corresponding class with minimum residual
    correct_count = 0;
    [value, index] = min(residual, [], 2);
    for i = 1:m2
        if index(i) == tt_label(i)
            correct_count = correct_count + 1;
        end
    end
    correct_count
end
