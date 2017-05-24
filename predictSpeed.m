data = xlsread('data.xlsx','Sheet1','C35:S65');
inputData = data(:, 17:-1:1);
errors = zeros(31,10);
rate = zeros(31,10);
predict = zeros(31,10);
for i = 4:31
    i
    oneData = inputData(i, :)';
    for j = 1:20
        j
        [errors(i,j),rate(i,j),predict(i,j)] = narnetP(oneData,4);
                serror_min(i,1) = errors(i,j);
        serror_min(i,2) = predict(i,j);
        serror_min(i,3) = rate(i,j);
        if(abs(rate(i,j))<20)
            break;
        end
    end
end