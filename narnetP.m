function [ error, rate, predict ] = narnetP( trainSet, lag )
%NARNET Summary of this function goes here
%   Detailed explanation goes here

% �ýű�������NAR������Ԥ��
% ���ߣ�Macer��
%lag=3;    
% �Իع����

% xΪԭʼ���У���������

[xn,xnopt] = mapminmax(trainSet(1:end-1)');
%׼��������������
iinput=xn(1:end); n=length(iinput);
inputs=zeros(lag,n-lag);
for i=1:n-lag    
    inputs(:,i)=iinput(i:i+lag-1);
end

targets=xn(lag+1:end);

input_train = inputs;
output_train = targets;
%�ڵ����
inputnum=lag;
hiddennum=10;
outputnum=1;

%ѵ�����ݺ�Ԥ������
% input_train=input(1:1900,:)';
input_test=xn(end-lag+1:end);
% output_train=output(1:1900)';
output_test=trainSet(end);

%ѡ����������������ݹ�һ��
% [inputn,inputps]=mapminmax(input_train);
% [outputn,outputps]=mapminmax(output_train);
inputn = input_train;
outputn = output_train;
%��������
net=newff(inputn,outputn,hiddennum);

%% �Ŵ��㷨������ʼ��
maxgen=20;                         %��������������������
sizepop=10;                        %��Ⱥ��ģ
pcross=[0.3];                       %�������ѡ��0��1֮��
pmutation=[0.1];                    %�������ѡ��0��1֮��

%�ڵ�����
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

lenchrom=ones(1,numsum);        
bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %���ݷ�Χ

%------------------------------------------------------��Ⱥ��ʼ��--------------------------------------------------------
individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %����Ⱥ��Ϣ����Ϊһ���ṹ��
avgfitness=[];                      %ÿһ����Ⱥ��ƽ����Ӧ��
bestfitness=[];                     %ÿһ����Ⱥ�������Ӧ��
bestchrom=[];                       %��Ӧ����õ�Ⱦɫ��
%��ʼ����Ⱥ
for i=1:sizepop
    %�������һ����Ⱥ
    individuals.chrom(i,:)=Code(lenchrom,bound);    %���루binary��grey�ı�����Ϊһ��ʵ����float�ı�����Ϊһ��ʵ��������
    x=individuals.chrom(i,:);
    %������Ӧ��
    individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   %Ⱦɫ�����Ӧ��
end

%����õ�Ⱦɫ��
[bestfitness bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %��õ�Ⱦɫ��
avgfitness=sum(individuals.fitness)/sizepop; %Ⱦɫ���ƽ����Ӧ��
% ��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
trace=[avgfitness bestfitness]; 
 
%% ���������ѳ�ʼ��ֵ��Ȩֵ
% ������ʼ
for i=1:maxgen
    i
    % ѡ��
    individuals=Select(individuals,sizepop); 
    avgfitness=sum(individuals.fitness)/sizepop;
    %����
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
    % ����
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
    % ������Ӧ�� 
    for j=1:sizepop
        x=individuals.chrom(j,:); %����
        individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   
    end
    
  %�ҵ���С�������Ӧ�ȵ�Ⱦɫ�弰��������Ⱥ�е�λ��
    [newbestfitness,newbestindex]=min(individuals.fitness);
    [worestfitness,worestindex]=max(individuals.fitness);
    % ������һ�ν�������õ�Ⱦɫ��
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    individuals.chrom(worestindex,:)=bestchrom;
    individuals.fitness(worestindex)=bestfitness;
    
    avgfitness=sum(individuals.fitness)/sizepop;
    
    trace=[trace;avgfitness bestfitness]; %��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��

end
%% �Ŵ��㷨������� 
 figure(1)
[r c]=size(trace);
plot([1:r]',trace(:,2),'b--');
title(['��Ӧ������  ' '��ֹ������' num2str(maxgen)]);
xlabel('��������');ylabel('��Ӧ��');
legend('ƽ����Ӧ��','�����Ӧ��');
disp('��Ӧ��                   ����');
x=bestchrom;

%% �����ų�ʼ��ֵȨֵ��������Ԥ��
% %���Ŵ��㷨�Ż���BP�������ֵԤ��
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP����ѵ��
%�����������
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

%����ѵ��
[net,per2]=train(net,inputn,outputn);

%% BP����Ԥ��
%���ݹ�һ��
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
%     %��������
%     hiddenLayerSize = 10; 
%     %���ز���Ԫ����
%     net = fitnet(hiddenLayerSize);
%     
%     
%     
%     % �������ϣ�����ѵ�������Ժ���֤���ݵı���
%       net.divideParam.trainRatio = 100/100;
%      net.divideParam.valRatio = 0/100;
%      net.divideParam.testRatio = 0/100;
%     net.trainParam.show = 50;  
%        net.trainParam.lr = 0.05;  
%        net.trainParam.mc = 0.9;  
%        net.trainParam.epochs = 8000;  
%        net.trainParam.goal = 0;
%     %ѵ������
%     [net,tr] = train(net,inputs,targets);
%     %% ����ͼ���ж���Ϻû�
%     yn=net(inputs);
%     errors=targets-yn;
%      figure, ploterrcorr(errors)                      
%     %������������������20lags��
% %     figure, parcorr(errors)                          
%     %����ƫ������%
% %     [h,pValue,stat,cValue]= lbqtest(errors)         
%     %Ljung��Box Q���飨20lags��
%      figure,plotresponse(con2seq(targets),con2seq(yn)) 
%     %��Ԥ���������ԭ����%
% %     figure, ploterrhist(errors)                      
%     %���ֱ��ͼ%
% %     figure, plotperform(tr)                          
%     %����½���%
%     
%     
%     % ����Ԥ������Ԥ�⼸��ʱ���
%     
%     fn=1;  
%     %Ԥ�ⲽ��Ϊfn��
% 
% f_in=iinput(n-lag+1:end);
% f_out=zeros(1,fn);  
% %Ԥ�����
% % �ಽԤ��ʱ���������ѭ�������������������
% for i=1:fn    
%     f_out(i)=net(f_in);    
%     f_in=[f_in(2:end);f_out(i)];
% end
% ����Ԥ��ͼ
% figure,plot(1:17,iinput,'b',17:18,[iinput(end),f_out],'r')
end

