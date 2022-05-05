%% Birth Certificate
% ===================================== %
% DATE OF BIRTH:    2022.03.31
% NAME OF FILE:     Exp2_GSEquivalent.m
% FILE OF PATH:     /.
% FUNC:
%   D2NN类，案例二：等效GS算法
%
%   
% =====================================


%% MNIST数据集做一个成像样本
load mnist.mat imgBin

% img = reshape(img(4444, :), [28,28,1]);
img = imgBin(4444, :);


%% D2NN参数
layerNum = 3;
unitSize = [30 30 28];
unitWidth = [5.1 5.1 6];

layerDistance = [0.001 70];

frequency = 26.8e9;

net = D2NN(layerNum, unitSize, unitWidth, layerDistance, frequency);


%% 数据集格式处理
% 平面波作为输入场
trainX = ones(unitSize(2),unitSize(2),1);

% 图像作为近场目标
trainY = reshape(img, 28, 28, []);

%% 训练参数
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',3000, ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs',10000, ...
    'MiniBatchSize',1, ...
    'VerboseFrequency', 1000);

net = net.trainD2NN(trainX, trainY, options, 'Regression');

% save EXP_2.mat net;

%% 结果显示
close all;

% D2NN结构
F = figure("Name", "D2NN"); clf; F.Position = [30,96,452,411];
net.plotD2NN({trainX}, 150, 1);


% 迭代过程中MSE曲线等
F = figure("Name", "Train Process"); clf; F.Position = [65,618,436,270];
net.plotTrainingRMSECurves();

% 近场结果
pY = net.netPredict(trainX, "3D");

% 混淆矩阵
F = figure("Name", "Confusion Matrix"); clf; F.Position = [522,175,1385,270];
subplot(1,3,1); imagesc(reshape(img, [28,28,1])); view(-90,-90); axis square; colorbar;
subplot(1,3,2); imagesc(abs(pY)); view(-90,-90); axis square; colorbar;
subplot(1,3,3); histogram(abs(pY(find(img(:))))); hold on; histogram(abs(pY(find(~img(:)))))


% 相位分布
F = figure("Name", "Phase Distribution at Plane"); clf;
F.Position = [1095,625,719,242];
net.plotPhase();



%% 量化
partition = linspace(-pi,pi,17); 
[~,quants] = quantiz(angle(net.M{2}), partition(2:end-1), partition(1:end-1));

net2 = net.setM(2, exp(1j * quants(:)));

% 相位分布
F = figure("Name", "Qua Phase Distribution at Plane"); clf;
F.Position = [1095,625,719,242];
net2.plotPhase();

% 近场结果
pY = net2.netPredict(trainX, "3D");

% 混淆矩阵
F = figure("Name", "Qua Confusion Matrix"); clf; F.Position = [522,175,1385,270];
subplot(1,3,1); imagesc(reshape(img, [28,28,1])); view(-90,-90); axis square; colorbar;
subplot(1,3,2); imagesc(abs(pY)); view(-90,-90); axis square; colorbar;
subplot(1,3,3); histogram(abs(pY(find(img(:))))); hold on; histogram(abs(pY(find(~img(:)))))
