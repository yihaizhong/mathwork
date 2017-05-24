function ret=select(individuals,sizepop) % 该函数用于进行选择操作
% individualsinput 种群信息
% sizepop input 种群规模
% ret output 选择后的新种群
%求适应度值倒数
fitness1 = 10./individuals.fitness;
%f i t n e s s 1 = 1 0 ./ i n d i v i d u a l s .f i t n e s s ; % i n d i v i d u a l s .f i t n e s s 为 个 体 适 应 度 值
%个体选择概率 
sumfitness=sum(fitness1); 
sumf=fitness1./sumfitness;
%采用轮盘赌法选择新个体
index= [];
for i=1:sizepop %sizepop为 种 群 数
    pick=rand; 
    while pick==0
    pick=rand;
    end
    for i=1:sizepop
        pick=pick-sumf(i); 
        if pick<0 
            index= [index i];
            break; 
        end
    end
end
%新种群 
individuals.chrom=individuals.chrom(index,:);
individuals.fitness=individuals.fitness(index); 
ret=individuals;

