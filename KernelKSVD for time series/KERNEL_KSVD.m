close all;
clear all;
clc;

%Importing the training and testing data
tr_dat = load('E:\graduate learning\UCRdataset\Coffee_TRAIN');
tt_dat = load('E:\graduate learning\UCRdataset\Coffee_TEST');
%Separating the data and labels
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
%      if tr_label(iter) == -1
%          tr_label(iter) = 1;
%      else
%         tr_label(iter) = 2;
%      end
% end
tt_label = tt_label + 1;
% for iter = 1:m2
%    if tt_label(iter) == -1
%        tt_label(iter) = 1;
%     else
%         tt_label(iter) = 2;
%     end
% end

%alpha, beta are parameters for TWED
alpha = 0;
beta = 0;
while alpha < 0.1
   beta = 0.2;
while beta < 1.1
for s = 1:3

iterNum = 40;                     %Number of loops for dictionary learning
classNum = 2;                     %Number of classes
sparsity = s;                     %Number for sparse coding
residual = zeros(classNum, m2);

for i = 1:classNum
    Dic = tr_dat(:, tr_label == i);

    dimension = size(Dic, 2);
    YTY = zeros(dimension, dimension);        %Calculation of K(Y, Y)
    for k1 = 1:dimension
        for k2 = 1:dimension
            if k1 == k2
                YTY(k1, k2) = TWED(Dic(:, k1), 0, alpha, beta);        %Here we applied linear kernel, Gaussian RBF is similar
				%YTY(k1, k2) = 1;
				
            else
                if k1 < k2
                    YTY(k1, k2) = 0.5 * (TWED(Dic(:, k1), 0, alpha, beta) + TWED(Dic(:, k2), 0, alpha, beta) - TWED(Dic(:, k1), Dic(:, k2), alpha, beta));
					%YTY(k1, k2) = exp(-1 * (ERP(Y(:, k1), Y(:, k2), 1) ^ 2) / lamda);
                    YTY(k2, k1) = YTY(k1, k2);
                end
            end
        end
    end

    [A, X] = KKSVD(YTY, sparsity, iterNum);
    for j = 1:m2
        z = tt_dat(:, j);
        zTY = zeros(1, dimension);                  %Calculation of K(z, Y)
        for k = 1:dimension
            zTY(1, k) = 0.5 * (TWED(z, 0, alpha, beta) + TWED(Dic(:, k), 0, alpha, beta) - TWED(z, Dic(:, k), alpha, beta));
        end
        [x, zTY, YTY] = KOMP_ONE(0, zTY, YTY, A, sparsity);
        residual(i, j) = TWED(z, 0, alpha, beta) - 2 * zTY * A * x + x' * A' * YTY * A * x;
    end
end

%Classification, finding the corresponding class with minimum residual
correct_count = 0;
[value, index] = min(abs(residual), [], 1);
for i = 1:m2
    if index(i) == tt_label(i)
        correct_count = correct_count + 1;
    end
end
disp(['alpha=   ',num2str(alpha), '     beta=   ',num2str(beta), '     sparsity=  ', num2str(sparsity)]);
correct_count
end
beta = beta + 0.1;
end
alpha = alpha + 0.2;
end
