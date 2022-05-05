%% Birth Certificate
% ===================================== %
% DATE OF BIRTH:    2022.03.31
% NAME OF FILE:     Exp4_2_AntiScatter_Class.m
% FILE OF PATH:     /.
% FUNC:
%   D2NN类，案例四：单类别样本抗散射
%
%
% =====================================


%% 导入数据集
load mnist.mat imgBin lab

img = imgBin(lab == 1, :);
img = img(1:2500, :)';

%% D2NN参数

layerNum = 5;
unitSize = [28 40 40 40 28];
unitWidth = [5 5.1 5.1 5.1 7.3];

layerDistance = [72 72 72 72];

frequency = 26.8e9;

net = D2NN(layerNum, unitSize, unitWidth, layerDistance, frequency);


%% 数据集格式处理
trainX = img(:, 1:2000);

trainY = trainX;

testX = img(:, 2001:end);
testY = testX;


%% 训练参数
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',10, ...
    'InitialLearnRate', 0.005, ...
    'MaxEpochs', 60, ...
    'MiniBatchSize', 200, ...
    'VerboseFrequency', 50, ...
    'ValidationData', {testX, testY});

net = net.trainD2NN(trainX, trainY, options, 'Regression');

% save EXP_4_2.mat net;
%% 结果显示
close all;

% 结构
F = figure("Name", "3D Structure"); clf; F.Position = [10,590,436,270];
net.plotD2NN({trainX(:,randi([1, size(trainX,2)], 1))}, 204,1);

% 迭代过程中MSE曲线等
F = figure("Name", "Train Process"); clf; F.Position = [550,590,436,270];
net.plotTrainingRMSECurves();

% 强度数据
F = figure("Name", "Target Plane Intensity Distribution"); clf; F.Position = [82,197,555,231];
pY = net.netPredict(trainX, "3D");
ind = randi([1 size(pY,3)], [1 4]);
for ii = 1:4
    subplot(2,4,ii); imagesc(reshape(trainX(:,ind(ii)), [28 28]))
    subplot(2,4,ii+4); imagesc(abs(pY(:,:,ind(ii))).^2);
end

% 验证集强度数据
F = figure("Name", "Test Target Plane Intensity Distribution"); clf; F.Position = [82,197,555,231];
pY = net.netPredict(testX, "3D");
ind = randi([1 size(pY,3)], [1 2]);
for ii = 1:2
    subplot(2,2,ii); imagesc(reshape(trainX(:,ind(ii)), [28 28]));
    subplot(2,2,ii+2); imagesc(abs(pY(:,:,ind(ii))).^2);
end

% 相位分布
F = figure("Name", "Phase Distribution at Plane"); clf;
F.Position = [650,197,555,231];
net.plotPhase();

