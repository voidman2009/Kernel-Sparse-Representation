%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is invoked by KKSVD(). It is utilized to double sparse coding
% the input signal z, where Y is the original training sample matrix,  
% YTY(i, j), zTY represent K(Yi, Yj), K(z, Yt) respectively. A is the matrix 
%where Phi(D) = Phi(Y) * A. During dictionary learning flag should be 0
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [x, zTY, YTY] = KOMP_ONE(flag, zTY, YTY, A, sparsity)

dimension = size(YTY, 2);
x = zeros(dimension, 1);

zk = zeros(dimension, 1);
index = zeros(sparsity, 1);
At = [];
for nIter = 1:sparsity
    proj = (zTY - zk' * YTY) * A;
    [value, pos] = sort(abs(proj), 2, 'descend');
    if flag == pos(1)
        index(nIter) = pos(2);
    else
        index(nIter) = pos(1);
    end
    At = [At, A(:, index(nIter))];
    K_s = At' * YTY * At;
    %If K_s is not semi positive definite, all negative eigenvalues 
    %need to be wiped off.
    [vectors, values] = eig(K_s);
    if(size(find(values < 0), 1) ~= 0)
        for i = 1:size(values, 1)
            if(values(i, i) < 0)
                values(i, i) = 0;
            end
        end
        K_s = vectors * values * vectors';
    end
    xt = pinv(K_s) * (zTY * At)';
    zk = At * xt;
    A(:, index(nIter)) = zeros(dimension, 1);
end
x(index(1:nIter)) = xt;

