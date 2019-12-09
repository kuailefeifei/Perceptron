%% Generation of TrainData and TestData
clc
clear all
x1 = rand(1,50);
x2 = rand(1,40);
x3 = rand(1,40);
w1_true = 0.85;
w2_true = 0.8;
w3_true = 0.5;
DataSet = zeros(numel(x1)*numel(x2)*numel(x3),3);
l = 1;
for i = 1:numel(x1)
    for j = 1:numel(x2)
        for k = 1:numel(x3)
            DataSet(l,1) = x1(i);
            DataSet(l,2) = x2(j);
            DataSet(l,3) = x3(k);
            DataSet(l,4) = w1_true*x1(i) + w2_true*x2(j) + w3_true*x3(k);
            l = l + 1;
        end
    end
end

samp = randperm(size(DataSet,1),round(size(DataSet,1)*0.7));
TrainData = DataSet(samp,:);
DataSet(samp,:) = [];
TestData = DataSet;

%% Initialization of weight factors and learning rate
w1_init = rand(1,1);
w2_init = rand(1,1);
w3_init = rand(1,1);
w1_delta = 0;
w2_delta = 0;
w3_delta = 0;
learn_rate = 0.01;

%% Start of learning
w1 = w1_init;
w2 = w2_init;
w3 = w2_init;

for i = 1:size(TrainData,1)
    out_pred = w1*TrainData(i,1) + w2*TrainData(i,2) + w3*TrainData(i,3);
    error(i) = TrainData(i,4) - out_pred;
    w1_delta = learn_rate*error(i)*TrainData(i,1);
    w2_delta = learn_rate*error(i)*TrainData(i,2);
    w3_delta = learn_rate*error(i)*TrainData(i,3);
    w1 = w1 + w1_delta;
    w2 = w2 + w2_delta;
    w3 = w3 + w3_delta;
end

figure
title('training process')
xlabel('calculation step')
ylabel('error')
plot(1:size(TrainData,1),abs(error))

%% evaluation of modell
rms = 0;

for i = 1:size(TestData,1)
    out_pred = w1*TestData(i,1) + w2*TestData(i,2) + w3*TestData(i,3);
    rms = rms + (TestData(i,4) - out_pred).^2;
end
weight_name = ['w1';'w2';'w3'];
true_value = [w1_true;w2_true;w3_true];
pred_value = [w1;w2;w3];
T = table(weight_name,true_value,pred_value)
rms = rms / size(TestData,1)

            