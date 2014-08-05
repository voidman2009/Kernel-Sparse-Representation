%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Kernel KSVD algorithm, the detail is given in the original paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A, X] = KKSVD(YTY, sparsity, iterNum)

dimension = size(YTY, 2);
A = eye(dimension);                   %Initialization of the dictionary
%During dictionary learning phase, YTY must be semi positive definite
[Vecs, Vals] = eig(YTY);
if(size(find(Vals < 0), 1) ~= 0)
    for i = 1:size(Vals, 1)
        if(Vals(i, i) < 0)
            Vals(i, i) = 0;
        end
    end
    YTY = Vecs * Vals * Vecs';
end

for iTer = 1:iterNum
    %sparse coding phase
    X = [];
    for i = 1:dimension
        [x, yTY, YTY] = KOMP_ONE(i, YTY(i, :), YTY, A, sparsity);
        X = [X, x];
    end
    %dictionary learning phase
    for i = 1:dimension
        
        xT = X(i, :);
        w = find(xT);
        if ~isempty(w)
            Omega = zeros(dimension, length(w));
            for j = 1:length(w)
                Omega(w(j), j) = 1;
            end
            E_k = eye(dimension) - A * X + A(:, i) * xT;
            E_R = E_k * Omega;
            if E_R == 0
                break;
            end
            [U, S, V] = svd(E_R' * YTY * E_R);
            A(:, i) = E_R * U(:, 1)/sqrt(S(1, 1));
            x_R = sqrt(S(1, 1)) * U(:, 1)';
            X(i, w) = x_R;
        end
    end
end