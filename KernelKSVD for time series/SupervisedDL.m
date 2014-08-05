%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Kernel LC_KSVD algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
%If the label are not starting at 0, then formalize it
tr_label = tr_label + 1;
% for iter = 1:m1
%    if tr_label(iter) == -1
%        tr_label(iter) = 1;
%    else
%       tr_label(iter) = 2;
%    end
% end
tt_label = tt_label + 1;
% for iter = 1:m2
%   if tt_label(iter) == -1
%       tt_label(iter) = 1;
%    else
%        tt_label(iter) = 2;
%    end
% end

tr_label
classNum = 2;                 %Number of classes
lamda = n1^2;                 %Gaussian RBF parameter
Q = zeros(m1, m1);
H = zeros(classNum, m1);
Q_class = [];
H_class = [];
classID = 1;
for i = 1:m1
    %Calculating values of Qi
    for j = 1:m1
        if tr_label(j) == tr_label(i)
            Q(j, i) = 1;
        end
    end
    %Calculating Hi
    H(tr_label(i), i) = 1;
    %if classID == tr_label(i)
    %    Q_class = [Q_class, Q(:, i)];
    %    H_class = [H_class, H(:, i)];
    %    classID = classID + 1;
    %end
end

%K(M, M)   The same as YTY in Kernel KSVD
MTM = zeros(m1, m1);
for k1 = 1:m1
    for k2 = 1:m1
        if k1 == k2
            %MTM(k1, k2) = ERP(tr_dat(:, k1), 0, 0);         %Linear kernel
            MTM(k1, k2) = 1;                                 %Gaussian RBF kernel
        else
            if k1 < k2
                %MTM(k1, k2) = 0.5 * (ERP(tr_dat(:, k1), 0, 0) + ERP(tr_dat(:, k2), 0, 0) - ERP(tr_dat(:, k1), tr_dat(:, k2), 1));
                MTM(k1, k2) = exp(-1 * (ERP(tr_dat(:, k1), tr_dat(:, k2), 1) ^ 2) / lamda);
                MTM(k2, k1) = MTM(k1, k2);
            end
        end
    end
end

%a1 is the same as alpha, b1 is the same as beta
a1 = 0;
while a1 <= 0.5
b1 = 0.01;
while b1 <= 0.51
for s1 = 1:3

sparsity = s1;
alpha = a1;             %LC_KSVD parameter
beta = b1;              %LC_KSVD parameter
iterNum = 100;          %Iteration of dictionary learning


%Computing YTY, after 3 items combine
YTY = zeros(m1, m1);
for k1 = 1:m1
    for k2 = 1:m1
        YTY(k1, k2) = MTM(k1, k2) + alpha * Q(:, k1)' * Q(:, k2) + beta * H(:, k1)' * H(:, k2);
    end
end

[B, X] = KKSVD(YTY, sparsity, iterNum);
W = H * B;
correct_count = 0;
X = [];

%Classification
for i = 1:m2
    %K(z, Y)
    zTY = zeros(1, m1);
    for j = 1:m1
        %zTY(1, j) = 0.5 * (ERP(tt_dat(:, i), 0, 0) + ERP(tr_dat(:, j),0, 0) - ERP(tt_dat(:, i), tr_dat(:, j), 1));
        zTY(1, j) = exp(-1 * (ERP(tt_dat(:, i), tr_dat(:, j), 1) ^ 2) / lamda);
    end
    [x] = KOMP_ONE(0, zTY, MTM, B, sparsity);
    [value, pos] = max(W * x);
    X = [X,x];
    if pos(1) == tt_label(i)
       correct_count = correct_count + 1;
    end
end

disp(['alpha=   ',num2str(alpha), '     beta=   ',num2str(beta), '     sparsity=  ', num2str(sparsity)]);
correct_count

end
b1 = b1 + 0.1;
end
a1 = a1 + 0.02;
end








