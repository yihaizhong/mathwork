errors = zeros(31,10);
rate = zeros(31,10);
predict = zeros(31,10);
for i = 1:31
    oneData = inputData(i, :)';
    for j = 1:10
        [errors(i,j),rate(i,j),predict(i,j)] = narnetP(oneData,4);
    end
end