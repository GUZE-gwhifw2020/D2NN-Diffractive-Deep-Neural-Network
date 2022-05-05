%% Birth Certificate
% ===================================== %
% DATE OF BIRTH:    2022.03.31
% NAME OF FILE:     Exp4_1_AntiScatter_Single.m
% FILE OF PATH:     /.
% FUNC:
%   D2NN类，案例四：单样本抗散射
%
%
% =====================================


%% 导入数据集
load mnist.mat imgBin

img = reshape(imgBin(54321,:), [28 28]);
%% D2NN参数

layerNum = 4;
unitSize = [28 28 28 28];
unitWidth = [5.1 5.1 5.1 5.1];

layerDistance = [70 70 70];

frequency = 26.8e9;

net = D2NN(layerNum, unitSize, unitWidth, layerDistance, frequency);


%% 数据集格式处理
trainX = img;

trainY = trainX;

%% 训练参数
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',300, ...
    'InitialLearnRate', 0.1, ...
    'MaxEpochs',1000, ...
    'MiniBatchSize',1, ...
    'VerboseFrequency', 50);

net = net.trainD2NN(trainX, trainY, options, 'Regression');


% save EXP_4_1.mat net
%% 结果显示
close all;

% 迭代过程中MSE曲线等
F = figure("Name", "Train Process"); clf; F.Position = [241,539,436,270];
net.plotTrainingRMSECurves();

% 强度数据
F = figure("Name", "Target Plane Intensity Distribution"); clf;
F.Position = [769,606,555,231];
pY = net.netPredict(trainX, "3D");
subplot(1,2,1); imagesc(trainX)
subplot(1,2,2); imagesc(abs(pY).^2);

% 相位分布
F = figure("Name", "Phase Distribution at Plane"); clf;
F.Position = [47,138,703,298];
net.plotPhase();

% 3D 结构
F = figure("Name", "D2NN Network"); clf;
F.Position = [763,76,560,420];
net.plotD2NN({trainX}, 140, 1);
