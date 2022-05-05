%% Birth Certificate
% ===================================== %
% DATE OF BIRTH:    2022.03.31
% NAME OF FILE:     Exp1_1_4MNISTClassify.m
% FILE OF PATH:     /.
% FUNC:
%   D2NN类，案例一：(实验一)4类MNIST数据集识别
%
%
% =====================================


%% 导入数据集
load mnist.mat imgBin lab

img = imgBin(ismember(lab, [0,1,2,4]), :);
lab = lab(ismember(lab, [0,1,2,4]));


%% D2NN参数

layerNum = 4;
unitSize = [28 22 22 2];
unitWidth = [3.2 5.1 5.1 65];

layerDistance = [72 72 72];

frequency = 26.8e9;

net = D2NN(layerNum, unitSize, unitWidth, layerDistance, frequency);


%% 数据集格式处理
trainX = img(1:20000, :)';

cateY = unique(lab);
trainY = double(lab(1:20000)' == cateY);
testX = img(20001:end, :)';
testY = double(lab(20001:end)' == cateY);
lab = lab(1:20000);

%% 训练参数

options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',20, ...
    'InitialLearnRate', 0.1, ...
    'GradientDecayFactor', 0.9, ...
    'SquaredGradientDecayFactor', 0.999, ...
    'MiniBatchSize', 10000, ...
    'MaxEpochs',100, ...
    'VerboseFrequency', 25, ...
    'ValidationData', {testX, testY});

net = net.trainD2NN(trainX, trainY, options, 'Classification');

% save EXP_1_1.mat net;
%%
% 全样本预测结果
pY = net.netPredict(reshape(trainX, 784, []));
% 最大标签
[~,indTemp] = max(abs(pY),[],1);

%% 结果显示
close all;

% D2NN结构
F = figure("Name", "D2NN"); clf; F.Position = [30,96,452,411];
net.plotD2NN({trainX(:,8333)}, 112, 1); % 2-2333,0-3333,4-4333,1-8333

% 迭代过程中MSE曲线等
F = figure("Name", "Train ProcessRMSE"); clf; F.Position = [65,618,436,270];
net.plotTrainingRMSECurves();

% 迭代过程中MSE曲线等
F = figure("Name", "Train ProcessAccuracy"); clf; F.Position = [65,618,436,270];
net.plotTrainingAccuCurves();


% 混淆矩阵
F = figure("Name", "Confusion Matrix"); clf; F.Position = [564,610,484,274];
confusionchart(lab, cateY(indTemp), RowSummary="row-normalized");
nnz(lab == cateY(indTemp)) / length(lab)


% 强度数据
F = figure("Name", "Target Plane Intensity Distribution"); clf;
F.Position = [522,175,1385,270];
for ii = 1:4
    subplot(1,4,ii); hold on;
    P = abs(pY(:, lab == cateY(ii)));
    for jj = 1:4; histogram(P(jj,:)); end
    hold off;
end

% 相位分布
F = figure("Name", "Phase Distribution at Plane"); clf;
F.Position = [1095,625,719,242];
net.plotPhase();