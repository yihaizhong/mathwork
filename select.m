function ret=select(individuals,sizepop) % �ú������ڽ���ѡ�����
% individualsinput ��Ⱥ��Ϣ
% sizepop input ��Ⱥ��ģ
% ret output ѡ��������Ⱥ
%����Ӧ��ֵ����
fitness1 = 10./individuals.fitness;
%f i t n e s s 1 = 1 0 ./ i n d i v i d u a l s .f i t n e s s ; % i n d i v i d u a l s .f i t n e s s Ϊ �� �� �� Ӧ �� ֵ
%����ѡ����� 
sumfitness=sum(fitness1); 
sumf=fitness1./sumfitness;
%�������̶ķ�ѡ���¸���
index= [];
for i=1:sizepop %sizepopΪ �� Ⱥ ��
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
%����Ⱥ 
individuals.chrom=individuals.chrom(index,:);
individuals.fitness=individuals.fitness(index); 
ret=individuals;

