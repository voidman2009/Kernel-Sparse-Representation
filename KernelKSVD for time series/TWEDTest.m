%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TWED test: Testing program for function TWED()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;
clc;

tr_dat = load('E:\data\SwedishLeaf_TRAIN');
tt_dat = load('E:\data\SwedishLeaf_TEST');
tr_label = tr_dat(:,1);
tt_label = tt_dat(:,1);
[m1,n1] = size(tr_dat);
tr_dat = tr_dat(1:m1,2:n1)';
[m2,n2] = size(tt_dat);
tt_dat = tt_dat(1:m2,2:n2)';

tr_dat = zscore(tr_dat);
tt_dat = zscore(tt_dat);
%tr_label = tr_label + 1;
% for iter = 1:m1
%   if tr_label(iter) == -1
%         tr_label(iter) = 1;
%     else
%        tr_label(iter) = 2;
%     end
% end
%tt_label = tt_label + 1;
% for iter = 1:m2
%     if tt_label(iter) == -1
%         tt_label(iter) = 1;
%     else
%         tt_label(iter) = 2;
%     end
% end

tr_label
alpha = 0.1;
beta = 0;
while alpha <= 1 
beta = 0;
while beta <= 1

classNum = 15;             %Number of classes
correct_count = 0;         %Number of test samples being correctly classified
for i = 1:m2
    residual = zeros(m1, 1);
    for j = 1:m1
        residual(j, 1) = TWED(tt_dat(:, i), tr_dat(:, j), alpha, beta);   %Calculating the distance between the test sample and every training sample
    end
    [values, index] = min(residual, [], 1);       %1NN-classifier
    if tr_label(index) == tt_label(i)
        correct_count = correct_count + 1;
    end
end
disp(['Alpha = ',num2str(alpha), '       Beta = ',num2str(beta)]);
correct_count
beta = beta + 0.1;
end
alpha = alpha + 0.1;
end