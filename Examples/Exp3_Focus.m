%% Birth Certificate
% ===================================== %
% DATE OF BIRTH:    2022.03.31
% NAME OF FILE:     Exp3_Focus.m
% FILE OF PATH:     /.
% FUNC:
%   D2NN类，案例三：聚焦点
%   4聚焦点，场强度分别为[1,2,3,4]
%   
% =====================================
close all;

%% 斑点
img = zeros(28);
img(7,7) = 1;
img(7,21) = 2;
img(21,7) = 3;
img(21,21) = 4;

%% D2NN参数

layerNum = 3;
unitSize = [40 40 28];
unitWidth = [10 10 20];

layerDistance = [0.01 100];

frequency = 11.6e9;

net = D2NN(layerNum, unitSize, unitWidth, layerDistance, frequency);


%% 数据集格式处理
% 平面波作为输入场
trainX = ones(40,40,1);

% 图像作为近场目标
trainY = reshape(img, 28, 28, []);

%% 训练参数
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',1000, ...
    'InitialLearnRate', 0.2, ...
    'MaxEpochs',10000, ...
    'MiniBatchSize',1, ...
    'VerboseFrequency', 1e9);

net = net.trainD2NN(trainX, trainY, options, 'Regression');

% save EXP_3_1.mat net
%% 结果显示
close all;

% 迭代过程中MSE曲线等
F = figure("Name", "Train Process"); clf; F.Position = [101,590,436,270];
net.plotTrainingRMSECurves();


% 近场结果
pY = net.netPredict(trainX, "3D");

F = figure("Name", "Confusion Matrix"); clf; F.Position = [101,590,436,270];
subplot(1,2,1); imagesc(reshape(img, [28,28,1])); axis square; colorbar;
subplot(1,2,2); imagesc(abs(pY)); axis square; colorbar;


% 相位分布
F = figure("Name", "Phase Distribution at Plane"); clf;
F.Position = [580,555,723,303];
net.plotPhase();

% D2NN结构
F = figure("Name", "D2NN"); clf; F.Position = [30,96,452,411];
net.plotD2NN({trainX}, 400, 1);

