%% Birth Certificate
% ===================================== %
% DATE OF BIRTH:    2022.04.10
% NAME OF FILE:     Exp0_CSTTest.m
% FILE OF PATH:     /.
% FUNC:
%   D2NN类，案例零
%   CSTFileWrite函数测试
%
%
% =====================================

%% D2NN参数

layerNum = 3;
unitSize = [18 18 18];
unitWidth = [10 10 10];

layerDistance = [0.01 50];

frequency = 11.6e9;

net = D2NN(layerNum, unitSize, unitWidth, layerDistance, frequency);


%% 数据集格式处理
trainX = ones(18);

%% 训练参数

% 不训练，直接赋值棋盘状1bit编码
c = mod(1:6,2);
gridPattern = kron(c == c', ones(3));
gridPattern = gridPattern(:);
net = net.setM(2, exp(1j * pi * gridPattern));

%% 结果显示
close all;

% D2NN结构
F = figure("Name", "D2NN"); clf; F.Position = [30,96,452,411];
net.plotD2NN({trainX});
colorbar

% 相位分布
F = figure("Name", "Phase Distribution at Plane"); clf;
F.Position = [1095,625,719,242];
net.plotPhase();

%%
load ../pa.mat paF
net.CSTFileWrite('Exp0_gridPattern_3_18.txt', ...
    @(x) paF.pha2Dx(x+pi, paF), ...
    "补充信息")
