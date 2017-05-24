data = xlsread('data.xlsx','Sheet1','C2:T32');
inputData = data(:, 18:-1:1);
errors = zeros(31,10);
rate = zeros(31,10);
predict = zeros(31,10);
for i = 7:31
    i
    oneData = inputData(i, :)';
    for j = 1:100;
        j
        [errors(i,j),rate(i,j),predict(i,j)] = narnetP(oneData,3);
        error_min(i,1) = errors(i,j);
        error_min(i,2) = predict(i,j);
        error_min(i,3) = rate(i,j);
        if(abs(rate(i,j))<5)
            break;
        end
    end
end