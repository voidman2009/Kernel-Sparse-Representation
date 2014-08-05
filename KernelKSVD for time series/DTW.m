%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Computing DTW distance between sample Xi and Xj, where Xi, Xj are column 
%vectors. If the flag equals 0, it means Xj is 0 vector, otherwise call the
%dynamic programming procedure to calculate the distance
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [distance] = DTW(Xi, Xj, flag)

[r1] = size(Xi, 1);
[r2] = size(Xj, 1);
M = zeros(r1, r2);
distance = 0;
if flag == 0
    for i = 1:r1
        distance = distance + abs(Xi(i));
    end
	return;
end

for i = 1:r1
    for j = 1:r2 
       M(i, j) = abs(Xi(i) - Xj(j));
    end
end

[D] = dp(M);
distance = D(r1, r2);