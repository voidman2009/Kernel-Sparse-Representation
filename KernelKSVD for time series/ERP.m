%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Computing ERP distance between sample Xi and Xj, where Xi, Xj are column 
%vectors. If the flag equals 0, it means Xj is 0 vector, otherwise call the
%dynamic programming procedure to calculate the distance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [distance] = ERP(Xi, Xj, flag)

dimension = size(Xi, 1);
distance = 0;
if 0 == flag
    for i = 1:dimension
        distance = distance + abs(Xi(i));
    end
else
    D = zeros(dimension + 1, dimension + 1);
    D(1, :) = NaN;
    D(:, 1) = NaN;
    D(1, 1) = 0;
    
    for i = 1:dimension
        for j = 1:dimension
            D(i + 1, j + 1) = min([D(i, j) + abs(Xi(i) - Xj(j)), D(i + 1, j) + abs(Xj(j)), D(i, j + 1) + abs(Xi(i))]);
        end
    end
    distance = D(dimension + 1, dimension + 1);
end
