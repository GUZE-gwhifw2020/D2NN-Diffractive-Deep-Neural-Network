%% Birth Certificate
% ===================================== %
% DATE OF BIRTH:    2022.04.01
% NAME OF FILE:     Exp1_3_10MNISTClassify_LargeScale.m
% FILE OF PATH:     /.
% FUNC:
%   D2NN类，案例一：(实验三)9类MNIST数据集识别
%
%
% =====================================
close all;

%% 导入数据集
load mnist.mat imgBin lab

img = imgBin(ismember(lab, [0,1,2,3,5,6,7,8,9]), :);
lab = lab(ismember(lab, [0,1,2,3,5,6,7,8,9]));

%% D2NN参数

layerNum = 6;
unitSize = [28 40 40 40 40 3];
unitWidth = [4.5 5.1 5.1 5.1 5.1 80];

layerDistance = [72 72 72 72 72];

frequency = 26.8e9;

net = D2NN(layerNum, unitSize, unitWidth, layerDistance, frequency);


%% 数据集格式处理
trainX = img(1:50000, :)';

cateY = unique(lab);
trainY = double(lab(1:50000)' == cateY);
testX = img(50001:end, :)';
testY = double(lab(50001:end)' == cateY);
lab = lab(1:50000);

%% 训练参数
options = trainingOptions('adam', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',4, ...
    'InitialLearnRate', 0.15, ...
    'MaxEpochs',20, ...
    'MiniBatchSize', 1000, ...
    'GradientDecayFactor', 0.9, ...
    'SquaredGradientDecayFactor', 0.999, ...
    'ValidationData', {testX, testY}, ...
    'VerboseFrequency', 400);

net = net.trainD2NN(trainX, trainY, options, 'Classification');

% save EXP_1_3.mat net
%%
tic
% 全样本预测结果
pY = net.netPredict(reshape(trainX, 784, []));
toc

%% 结果显示
% 结构
F = figure("Name", "Train Process"); clf; F.Position = [10,590,436,270];
ind = randperm(40000,1);
net.plotD2NN({trainX(:, ind)}, 250, 1);

% 迭代过程中MSE曲线等
F = figure("Name", "Train Process"); clf; F.Position = [10,590,436,270];
net.plotTrainingRMSECurves();

% 迭代过程中MSE曲线等
F = figure("Name", "Train Process"); clf; F.Position = [10,590,436,270];
net.plotTrainingAccuCurves();

% 最大标签
[~,indTemp] = max(abs(pY),[],1);

% 混淆矩阵
confusionmat(lab, cateY(indTemp))
nnz(lab == cateY(indTemp)) / numel(lab)
F = figure("Name", "Confusion Matrix"); clf; F.Position = [450,590,436,270];
confusionchart(lab, cateY(indTemp), RowSummary="row-normalized");


% 强度数据
F = figure("Name", "Target Plane Intensity Distribution"); clf;
F.Position = [82,179,1458,249];
for ii = 1:4
    subplot(1,4,ii); hold on;
    P = abs(pY(:, lab == cateY(ii)));
    for jj = 1:5; histogram(P(jj,:)); end
    hold off;
end

% 相位分布
F = figure("Name", "Phase Distribution at Plane"); clf;
F.Position = [900,555,723,303];
net.plotPhase();

