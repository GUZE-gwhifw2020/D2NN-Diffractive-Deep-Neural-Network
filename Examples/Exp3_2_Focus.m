%% Birth Certificate
% ===================================== %
% DATE OF BIRTH:    2022.05.04
% NAME OF FILE:     Exp3_2_Focus.m
% FILE OF PATH:     /.
% FUNC:
%   D2NN类，案例三：聚焦点
%   采用Classification方法（定稿，不要动）
%   
% =====================================
close all;

%% D2NN参数

layerNum = 3;
unitSize = [24 24 4];
unitWidth = [5.1 5.1 32];

layerDistance = [0.001 70];

frequency = 26.8e9;

net = D2NN(layerNum, unitSize, unitWidth, layerDistance, frequency);


%% 数据集格式处理
% 平面波作为输入场
trainX = ones(unitSize(1), unitSize(1), 1);

% 图像作为近场目标
trainY = zeros(4);
trainY(2,2) = 1;
trainY(3,3) = 1;
trainY(4,4) = 1;


%% 训练参数
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',1000, ...
    'InitialLearnRate', 0.05, ...
    'MaxEpochs',10000, ...
    'MiniBatchSize',1, ...
    'VerboseFrequency', 1e9);

net = net.trainD2NN(trainX, trainY, options, 'Classification');

% save EXP_3_2.mat net trainX;
%% 结果显示
close all;

% 迭代过程中MSE曲线等
F = figure("Name", "Train Process"); clf; F.Position = [101,590,436,270];
net.plotTrainingRMSECurves();


% 近场结果
pY = net.netPredict(trainX, "3D");

% 相位分布
F = figure("Name", "Phase Distribution at Plane"); clf;
F.Position = [580,555,723,303];
net.plotPhase();

% D2NN结构
F = figure("Name", "D2NN"); clf; F.Position = [30,96,452,411];
net.plotD2NN({trainX}, 123, 1);

