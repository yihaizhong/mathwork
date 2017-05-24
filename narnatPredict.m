function [ net, tr, perf, e ] = narnatPredict( trainData, testData, inputDelays, hideLayerSize )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
net = narnet(1:inputDelays, hideLayerSize);
   % 避免过拟合，划分训练，测试和验证数据的比例
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
           net.trainParam.show = 50;  
       net.trainParam.lr = 0.05;  
       net.trainParam.mc = 0.9;  
       net.trainParam.epochs = 8000;  
       net.trainParam.goal = 1e-9;
[Xs,Xi,Ai,Ts] = preparets(net, {}, {}, trainData);
[net, tr] = train(net, Xs, Ts, Xi, Ai);
% view(net)
Y = net(Xs,Xi);
perf = perform(net,Ts,Y)

%net = closeloop(net);
yini = trainData(end-inputDelays+2:end);
yini = [yini 0];
% combine initial values and validation data 'yv'
net = removedelay(net);
% [xs,xis,ais,ts] = preparets(nets,{},{},targetSeries);
% ys = nets(xs,xis,ais);
[Xs,Xi,Ai] = preparets(net,{},{},yini);

% predict on validation data
predict = net(Xs,Xi,Ai);

% validation data
Yv = cell2mat(testData);
% prediction
Yp = cell2mat(predict);
% error
e = Yv - Yp;
e = Yp;

end

