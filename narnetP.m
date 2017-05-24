function [ error, rate, predict ] = narnetP( trainSet, lag )
%NARNET Summary of this function goes here
%   Detailed explanation goes here

% 该脚本用来做NAR神经网络预测
% 作者：Macer程
%lag=3;    
% 自回归阶数

% x为原始序列（行向量）

[xn,xnopt] = mapminmax(trainSet(1:end-1)');
%准备输入和输出数据
iinput=xn(1:end); n=length(iinput);
inputs=zeros(lag,n-lag);
for i=1:n-lag    
    inputs(:,i)=iinput(i:i+lag-1);
end

targets=xn(lag+1:end);

input_train = inputs;
output_train = targets;
%节点个数
inputnum=lag;
hiddennum=10;
outputnum=1;

%训练数据和预测数据
% input_train=input(1:1900,:)';
input_test=xn(end-lag+1:end);
% output_train=output(1:1900)';
output_test=trainSet(end);

%选连样本输入输出数据归一化
% [inputn,inputps]=mapminmax(input_train);
% [outputn,outputps]=mapminmax(output_train);
inputn = input_train;
outputn = output_train;
%构建网络
net=newff(inputn,outputn,hiddennum);

%% 遗传算法参数初始化
maxgen=20;                         %进化代数，即迭代次数
sizepop=10;                        %种群规模
pcross=[0.3];                       %交叉概率选择，0和1之间
pmutation=[0.1];                    %变异概率选择，0和1之间

%节点总数
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

lenchrom=ones(1,numsum);        
bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %数据范围

%------------------------------------------------------种群初始化--------------------------------------------------------
individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %将种群信息定义为一个结构体
avgfitness=[];                      %每一代种群的平均适应度
bestfitness=[];                     %每一代种群的最佳适应度
bestchrom=[];                       %适应度最好的染色体
%初始化种群
for i=1:sizepop
    %随机产生一个种群
    individuals.chrom(i,:)=Code(lenchrom,bound);    %编码（binary和grey的编码结果为一个实数，float的编码结果为一个实数向量）
    x=individuals.chrom(i,:);
    %计算适应度
    individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   %染色体的适应度
end

%找最好的染色体
[bestfitness bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %最好的染色体
avgfitness=sum(individuals.fitness)/sizepop; %染色体的平均适应度
% 记录每一代进化中最好的适应度和平均适应度
trace=[avgfitness bestfitness]; 
 
%% 迭代求解最佳初始阀值和权值
% 进化开始
for i=1:maxgen
    i
    % 选择
    individuals=Select(individuals,sizepop); 
    avgfitness=sum(individuals.fitness)/sizepop;
    %交叉
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
    % 变异
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
    % 计算适应度 
    for j=1:sizepop
        x=individuals.chrom(j,:); %解码
        individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   
    end
    
  %找到最小和最大适应度的染色体及它们在种群中的位置
    [newbestfitness,newbestindex]=min(individuals.fitness);
    [worestfitness,worestindex]=max(individuals.fitness);
    % 代替上一次进化中最好的染色体
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;
    individuals.fitness(worestindex)=bestfitness;
    
    avgfitness=sum(individuals.fitness)/sizepop;
    
    trace=[trace;avgfitness bestfitness]; %记录每一代进化中最好的适应度和平均适应度

end
%% 遗传算法结果分析 
 figure(1)
[r c]=size(trace);
plot([1:r]',trace(:,2),'b--');
title(['适应度曲线  ' '终止代数＝' num2str(maxgen)]);
xlabel('进化代数');ylabel('适应度');
legend('平均适应度','最佳适应度');
disp('适应度                   变量');
x=bestchrom;

%% 把最优初始阀值权值赋予网络预测
% %用遗传算法优化的BP网络进行值预测
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP网络训练
%网络进化参数
% net.trainParam.epochs=100;
% net.trainParam.lr=0.1;
%       net.divideParam.trainRatio = 70/100;
%      net.divideParam.valRatio = 15/100;
%      net.divideParam.testRatio = 15/100;
    net.trainParam.show = 50;  
       net.trainParam.lr = 0.05;  
       net.trainParam.mc = 0.9;  
       net.trainParam.epochs = 8000;  
       net.trainParam.goal = 0;
%net.trainParam.goal=0.00001;

%网络训练
[net,per2]=train(net,inputn,outputn);

%% BP网络预测
%数据归一化
% inputn_test=mapminmax('apply',input_test,inputps);

inputn_test = input_test';
an=sim(net,[inputn inputn_test]);
test_simu=mapminmax('reverse',an,xnopt);
tg = trainSet(lag+1:end-1)';
error=test_simu-[tg output_test];
rate = error./output_test*100;
rate = rate(end);
predict = test_simu(end);
error = error(end);
% setdemorandstream(12);
%     %创建网络
%     hiddenLayerSize = 10; 
%     %隐藏层神经元个数
%     net = fitnet(hiddenLayerSize);
%     
%     
%     
%     % 避免过拟合，划分训练，测试和验证数据的比例
%       net.divideParam.trainRatio = 100/100;
%      net.divideParam.valRatio = 0/100;
%      net.divideParam.testRatio = 0/100;
%     net.trainParam.show = 50;  
%        net.trainParam.lr = 0.05;  
%        net.trainParam.mc = 0.9;  
%        net.trainParam.epochs = 8000;  
%        net.trainParam.goal = 0;
%     %训练网络
%     [net,tr] = train(net,inputs,targets);
%     %% 根据图表判断拟合好坏
%     yn=net(inputs);
%     errors=targets-yn;
%      figure, ploterrcorr(errors)                      
%     %绘制误差的自相关情况（20lags）
% %     figure, parcorr(errors)                          
%     %绘制偏相关情况%
% %     [h,pValue,stat,cValue]= lbqtest(errors)         
%     %Ljung－Box Q检验（20lags）
%      figure,plotresponse(con2seq(targets),con2seq(yn)) 
%     %看预测的趋势与原趋势%
% %     figure, ploterrhist(errors)                      
%     %误差直方图%
% %     figure, plotperform(tr)                          
%     %误差下降线%
%     
%     
%     % 下面预测往后预测几个时间段
%     
%     fn=1;  
%     %预测步数为fn。
% 
% f_in=iinput(n-lag+1:end);
% f_out=zeros(1,fn);  
% %预测输出
% % 多步预测时，用下面的循环将网络输出重新输入
% for i=1:fn    
%     f_out(i)=net(f_in);    
%     f_in=[f_in(2:end);f_out(i)];
% end
% 画出预测图
% figure,plot(1:17,iinput,'b',17:18,[iinput(end),f_out],'r')
end

