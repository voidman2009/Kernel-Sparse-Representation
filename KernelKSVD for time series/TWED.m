%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Computing the TWED distance between two samples Xi and Xj, where Xi, Xj are
%column vectors, alpha is used to control the impact of time shifting, beta
%is set to control the impact of difference of values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [distance] = TWED(Xi, Xj, alpha, beta)

gama = alpha;             %gama >= 0
lamda = beta;            %lamda >= 0
distance = 0;
dimension = size(Xi, 1);

D = zeros(dimension + 1, dimension + 1);
D(1, :) = NaN;
D(:, 1) = NaN;
D(1, 1) = 0;
Si = zeros(dimension + 1, 1);
Sj = zeros(dimension + 1, 1);
Si(2:dimension + 1, 1) = Xi;
Sj(2:dimension + 1, 1) = Xj;

for i = 1:dimension
    for j = 1:dimension
        temp1 = abs(Si(i + 1) - Si(i)) + gama + lamda; 
        temp2 = abs(Si(i + 1) - Sj(j + 1)) + abs(Si(i) - Sj(j)) + 2 * gama * abs(i - j);
        temp3 = abs(Sj(j + 1) - Sj(j)) + gama + lamda;
        D(i + 1, j + 1) = min([D(i, j + 1) + temp1, D(i, j) + temp2, D(i + 1, j) + temp3]);
    end
end
distance = D(dimension + 1, dimension + 1);

end