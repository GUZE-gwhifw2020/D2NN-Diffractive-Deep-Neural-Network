classdef D2NN
    %D2NN Diffractive Deep Neural Network
    %   衍射深度神经网络 模型与训练算法
	%	Reference: < In situ optical backpropagation training of diffractive optical neural networks >


    properties (Constant, GetAccess = private)
        c0 = 299792458 * 1e3;   % 光速，单位mm/s
    end

    properties (SetAccess = immutable, GetAccess = public)  % 不可变类
        layerNum        % 层数（包括输入层与输出层）

        unitSize        % 每层单元一维边长个数，大小layerNum * 1
        unitNums        % 每层单元数，大小layerNum * 1，等于unitSize.^2
        unitWidth       % 每层单元边长，单位mm，大小layerNum * 1

        layerDistance   % 层间距离，单位mm，大小(layerNum-1) * 1

        frequency       % 频率，单位Hz


    end

    % 核心参数
    properties (SetAccess = private, GetAccess = public)
        M               % 调制矩阵，Diagnoal Diffractive Modulation Matrix
                        % cell类型，大小(layerNum) * 1
                        % 子元素大小（unitSize[k]^2 * 1）
                        % 第一个与最后一个元素为 ONES(?, 1)
                        % 这里原本为矩阵，现在改为列向量，乘法 * 变为 .*

    end

    %
    properties (SetAccess = immutable, GetAccess = protected)
        W               % 散射矩阵，Diffractive Weight Matrix，
                        % cell类型，大小(layerNum-1) * 1
                        % 子元素大小（unitSize[k-1]^2 * unitSize[k]^2）
    end

    % 网络参数
    properties (Constant, GetAccess = private)
        % 网络输出类型
        %   梯度计算公式gradientUpdate
        %   
        netTypeClassification = 0           % 分类
        netTypeRegression = 1               % 回归
    end

    properties (SetAccess = private, GetAccess = public)
        netType          % 网络类型

        trainingOptions  % 参考MATLAB内置机器学习的参数

        sampleNum        % 样本数

        mu               % 学习率（动态变化）
        
        iter             % 迭代次数

        trainStartTime      % 训练起始时间
        trainDuration       % 训练时间

        miniBatchRMSE       % Minibatch根均方差(Root-mean-squared-error)
        verboseRMSE         % 全数据集根均方差
        validationRMSE      % 验证数据集根均方差
        isValidationSet     % 是否存在验证数据集

        miniBatchAccu       % Minibatch准确率
        verboseAccu         % 全数据集准确率
        validationAccu      % 验证数据集准确率

    end

    % ADAM Gradient Algorithm 参数
    % https://ruder.io/optimizing-gradient-descent/index.html#adam
    properties (SetAccess = private, GetAccess = private)
        m               % Cell类型
        v               % Cell类型
        % beta          % 此参数在trainingOptions.GradientDecayFactor中
        % expsilon      % 此参数在trainingOptions.Epsilon中
    end

    properties (Constant, GetAccess = private)
        lizardTail = @(x) x(1:end-1);   % 字符串去尾函数
    end

    methods (Access = public)
        function obj = D2NN(layerNum, unitSize, unitWidth, layerDistance, frequency, radiationPattern0)
            %D2NN 构造此类的实例
            %   Input:
            %       layerNum            层数，包括输入与输出层
            %       unitSize            层阵列单元边长数
            %       unitWidth           单元边长（mm）
            %       layerDistance       层间距离（mm）
            %       frequency           频率（GHz）
            %       radiationPattern0   (缺省)方向图, 结构体
            %                           .THETA  0-45有效
            %                           .PHI    0-360有效(覆盖+z平面)    
            %                           .ABS    非dB值

            assert(length(unitSize) == layerNum, '单元数组与层数不一致');
            assert(length(unitWidth) == layerNum, '单元边长数组与层数不一致');
            assert(length(layerDistance) == layerNum-1, '间距数组与层数不一致');
            assert(layerDistance(1) >=0 && all(layerDistance(2:end)>0), '间距数组数值需大于0');


            obj.layerNum = layerNum;
            obj.unitSize = unitSize;
            obj.unitNums = unitSize.^2;
            obj.unitWidth = unitWidth;
            obj.layerDistance = layerDistance;

            obj.frequency = frequency;

            % 生成散射矩阵W
            obj.W = cell(obj.layerNum-1, 1);
            for ii = 1:(obj.layerNum-1)
                obj.W{ii} = obj.WGenerate(obj.unitWidth([ii ii+1]), obj.unitSize([ii ii+1]), obj.layerDistance(ii), obj.frequency);
            end

            % 初始化调制矩阵M
            obj.M = arrayfun(@(x) ones(x,1), obj.unitNums, 'UniformOutput', 0);

        end

        % = = = = = 核心函数 = = = = = %
        % ============================ %

        function obj = trainD2NN(obj, trainX, trainY, trainingOptions, inputNetType)
            %trainD2NN 基于样本集训练D2NN
            %   Input:
            %       trainX  数据集，大小 M(1) * M(1) * N 或 N(1) * N
            %       trainY  标签集，大小 M(e) * M(e) * N 或 N(end) * N
            %       trainingOptions 借用Deeping Learning Tollbox中TrainingOptionsADAM
            %       inputNetType    Regression-回归，Classification-分类

            % 网络类型参数
            if(strcmpi(inputNetType, 'Regression'))
                obj.netType = obj.netTypeRegression;
            elseif(strcmpi(inputNetType, 'Classification'))
                obj.netType = obj.netTypeClassification;
            else
                error("Error: inputNetType仅支持'Regression'与'Classification'。");
            end

            % 数据格式修正(assert函数内置在里面)
            [trainX, trainY] = obj.dataPreProcess(trainX, trainY);

            if(~isempty(trainingOptions.ValidationData))
                % 验证集数据预处理
                [testX, testY] = obj.dataPreProcess(trainingOptions.ValidationData{1}, ...
                    trainingOptions.ValidationData{2});

                % 删除trainingOptions中验证数据集
                trainingOptions.ValidationData = [];

                % 类中BOOL型变量写
                obj.isValidationSet = true;
            else
                obj.isValidationSet = false;
            end


            % 样本数
            obj.sampleNum = size(trainX, 2);

            % trainOpt复制过来
            obj.trainingOptions = trainingOptions;

            % 学习率
            obj.mu = obj.trainingOptions.InitialLearnRate;

            % 洗牌
            ind = randperm(obj.sampleNum);
            trainX = trainX(:,ind);
            trainY = trainY(:,ind);

            % 学习批次数
            batchNum = ceil(obj.sampleNum / obj.trainingOptions.MiniBatchSize);
            batchInd = (1:obj.trainingOptions.MiniBatchSize)' + ...
                round(linspace(1,obj.sampleNum-obj.trainingOptions.MiniBatchSize, batchNum));

            % RMSE数据集
            obj.miniBatchRMSE = zeros(obj.trainingOptions.MaxEpochs * batchNum, 1);
            obj.verboseRMSE = zeros(floor(obj.trainingOptions.MaxEpochs * batchNum / obj.trainingOptions.VerboseFrequency), 1);
            obj.validationRMSE = zeros(floor(obj.trainingOptions.MaxEpochs * batchNum / obj.trainingOptions.VerboseFrequency), 1);

            % Accuracy数据集(CLASSIFICATION下才有效)
            obj.miniBatchAccu = nan(obj.trainingOptions.MaxEpochs * batchNum, 1);
            obj.verboseAccu = nan(floor(obj.trainingOptions.MaxEpochs * batchNum / obj.trainingOptions.VerboseFrequency), 1);
            obj.validationAccu = nan(floor(obj.trainingOptions.MaxEpochs * batchNum / obj.trainingOptions.VerboseFrequency), 1);


            % ADAM动量项初始化
            obj.m = arrayfun(@(x) zeros(x,1), obj.unitNums, 'UniformOutput', 0);
            obj.v = arrayfun(@(x) zeros(x,1), obj.unitNums, 'UniformOutput', 0);

            % 训练迭代初始
            obj.iter = 1;
            obj.trainStartTime = datetime('now');
            fprintf("%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t%s\n", "Duration", ...
                "Epoch", "Batch", "Batch MSE", "Verbose MSE", "Validation MSE", "Mu", "ETA");

            % 验证集MSE数据占坑
            mseVadTemp = nan;


            %
            for epoch = 1:obj.trainingOptions.MaxEpochs

                % batchLearn
                for batch = 1:batchNum

                    % 梯度下降，更新每一层的调制矩阵
                    obj = obj.gradientUpdate(trainX(:,batchInd(:, batch)), trainY(:,batchInd(:, batch)));

                    % Verbose,显示环节,全数据集Loss
                    if(~mod(obj.iter, obj.trainingOptions.VerboseFrequency))
                        [mseTemp, accuTemp] = obj.lossCalFcn(trainX, trainY);
                        obj.verboseRMSE(obj.iter / obj.trainingOptions.VerboseFrequency) = mseTemp;
                        obj.verboseAccu(obj.iter / obj.trainingOptions.VerboseFrequency) = accuTemp;


                        % 验证集数据
                        if(obj.isValidationSet)
                            [mseVadTemp, accuVadTemp] = obj.lossCalFcn(testX, testY);
                            obj.validationRMSE(obj.iter / obj.trainingOptions.VerboseFrequency) = mseVadTemp;
                            obj.validationAccu(obj.iter / obj.trainingOptions.VerboseFrequency) = accuVadTemp;
                        end

                        % 显示
                        duaration = datetime('now') - obj.trainStartTime;
                        fprintf("%s\t%d(%d)\t%d(%d)\t%.3e\t%.3e\t%.3e\t%.3e\t%s\n", ...
                            duaration, epoch, obj.trainingOptions.MaxEpochs, ...
                            batch, batchNum, obj.miniBatchRMSE(obj.iter), mseTemp, mseVadTemp, obj.mu, ...
                            datestr(obj.trainStartTime + duaration / obj.iter * batchNum * obj.trainingOptions.MaxEpochs));
                    end

                    obj.iter = obj.iter + 1;
                end

                % 学习率逐epoch下降
                if(~mod(epoch, obj.trainingOptions.LearnRateDropPeriod))
                    obj.mu = obj.mu * obj.trainingOptions.LearnRateDropFactor;
                end
            end

            % 结束训练
            obj.trainDuration = datetime('now') - obj.trainStartTime;
        end


        % = = = = = 辅助函数 = = = = = %
        % ============================ %

        function [optMu,muArray,mseM] = optimumMuFind(obj, trainX, trainY, trainOpts, inputNetType, muArray, iterMax)
            %optimumMuFind 寻找最优optMu
            % Input:
            %   trainX, trainY      训练数据
            %   trainOpts           训练参数，其中mu不会被用到
            %   muArray             （不允许缺省）待比较学习率向量
            %   iterMax             （允许缺省）最大迭代次数，当小于一个Epoch时有用
            %
            %   注意：1. 此过程不会修改类成员obj.M等，仅用于测试
            %         2. 用到parfor
            %         3. Default Mu = logspace(log10(0.01),log10(3),64)
            %

            % 网络类型参数
            if(strcmpi(inputNetType, 'Regression'))
                obj.netType = obj.netTypeRegression;
            elseif(strcmpi(inputNetType, 'Classification'))
                obj.netType = obj.netTypeClassification;
            else
                error("Error: inputNetType仅支持'Regression'与'Classification'。");
            end


            % 数据格式修正(assert函数内置在里面)
            [trainX, trainY] = obj.dataPreProcess(trainX, trainY);

            ind = randperm(size(trainX, 2));
            trainX = trainX(:,ind);
            trainY = trainY(:,ind);

            % mu
            if(isempty(muArray))
                muArray = logspace(log10(0.01),log10(3),64);
            end

            % iterMax
            if(nargin == 6)
                iterMax = inf;
            end


            % 并行运算的D2NN准备
            % mseVerb = cell(length(muArray),1);
            mseMini = cell(length(muArray),1);
            lenMuArray = length(muArray);

            tic
            parfor ii = 1:length(muArray)
                options = trainOpts;
                options.InitialLearnRate = muArray(ii);

                D2NNCell = trainD2NNSimplify(obj, trainX, trainY, options, iterMax);
                % mseMini{ii} = D2NNCell.verboseRMSE;
                mseMini{ii} = D2NNCell.miniBatchRMSE;
                fprintf("\t批次：%02d(%02d)已完成。\n", ii, lenMuArray);
            end
            toc

            % 数据处理
            % mseV = cellfun(@(x) x(end), mseVerb);
            mseM = cellfun(@(x) x(end), mseMini);

            % [~, I1] = min(mseV);
            [~, I2] = min(mseM);

            optMu = muArray(I2);

            figure("Name", "Optimum Mu Find Result"); clf;
            loglog(muArray, mseM);
            xline(muArray(I2));
            grid on;

            xlabel('\mu');
            ylabel('RMSE at last iteration');

        end


        % = = = = = 预测函数 = = = = = %
        % ============================ %

        function [loss, accu] = lossCalFcn(obj, X, Y)
            %lossCalFcn 计算LOSS与准确率
            %   Input:
            %       X/Y     一个/多个样本与样本输出
            %   Output:
            %       Loss    RMSE值
            %       Accu    (仅对分类问题有效)样本集正确率
            %               对回归问题，默认返回Nan

            [X,Y] = obj.dataPreProcess(X,Y);

            U = obj.netPredict(X);
            O = U .* conj(U);

            switch(obj.netType)
                case(obj.netTypeClassification)
                    
                    % 分类问题, Loss = CrossEntropy(O,Y)
                    O = softmax(O);
                    loss = mean(-log(O(find(Y)))); %#ok<FNDSB> %(有待改进)

                    [~,ia] = max(O);
                    accu = mean(Y(ia + (0:size(Y,1):size(Y,1)*length(ia)-1)));

                case(obj.netTypeRegression)

                    % 基于标签（理论输出结果）对网络输出结果反校正
                    O = O ./ vecnorm(O) .* vecnorm(Y);

                    % 回归问题, Loss = |O-Y|_2^2
                    loss = mean(sum((O - Y).^2));

                    accu = nan;

            end

        end

        function y = netPredict(obj, x, OutputType)
            %netPredict 输入网络预测数据
            %   Input:
            %       x           一个/多个样本
            %       OutputType  样本输出维度类型
            %                   (默认) L * N, L为输出神经元数, N为样本数
            %                   "3D"   K * K * N, 其中L = K^2

            x = obj.dataPreProcess(x);
            y = x;

            % 传播通过网络，例如3层网络，仅需2层运算
            % 注意：最后一层为M为ONES列向量
            % 第i层a个元素，第(i+1)层b个元素。n个样本(a,b均为平方数)
            % Size: [b,n] = [b,1].*([b,a]*[a,n])

            for ii = 1:(obj.layerNum-1)
                y = obj.M{ii+1} .* (obj.W{ii} * y);
            end

            % 输出要求为3-D数据形式，进行拉伸
            if(nargin == 3)
                if(strcmp(OutputType, "3D"))
                    y = reshape(y, obj.unitSize(end), obj.unitSize(end), []);
                end
            end
        end


        % = = = = = 可视化函数 = = = = = %
        % =========================== == %

        function plotD2NN(obj, sample, Nnew, Wnew, Dnew)
            %plotD2NN 3D绘图，D2NN结构
            % Input:
            %   sample  Cell类型，长度为1或2
            %           长度为1时，认为是X，计算输出Y，绘制输入层和通过网络图像
            %           长度为2时，认为是{X,Y}，绘制输入层和输出层网络图像
            %           无此变量时，仅仅绘制结构
            %
            %   
            %   Nnew    修正最后一层单元数   (允许同Wnew一同缺省)
            %   Wnew    修正最后一层单元间距 (允许同Nnew一同缺省)
            %   Dnew    修正最后一层垂直距离 (允许缺省，默认为当前层间距)

            hold on;
            view([-16 11.5]);

            dis = [0 cumsum(obj.layerDistance)];

            % 逐层绘制中间层（与sample, Nnew, Dnew无关）
            color = zeros(obj.layerNum,3);
            color(1,:) = [1 0 0 ]; color(end,:) = [0 0 1];
            for ii = 2:obj.layerNum-1
                plotSingleLayer(obj.unitSize(ii), obj.unitWidth(ii), ...
                    dis(ii), color(ii,:));
            end

            % 样本绘制
            if(nargin == 1)

                % 输入、输出层内部线
                plotSingleLayer(obj.unitSize(1), obj.unitWidth(1), dis(1), color(1,:));
                plotSingleLayer(obj.unitSize(end), obj.unitWidth(end), dis(end), color(end,:));

                % 1个波长基准线
                lambda = obj.c0 / obj.frequency;
                line([0 0], [0 0], [0 lambda], 'LineWidth', 2);
                line([0 0], [0 lambda], [0 0], 'LineWidth', 2);
                line([-lambda 0], [0 0], [0 0], 'LineWidth', 2);


            elseif(nargin == 2)

                % 绘制样本
                switch length(sample)
                    case 1 % Y由网络计算提供
                    X = obj.dataPreProcess(sample{1});
                    Y = obj.netPredict(X);
                    case 2 % Y由用户提供
                    [X, Y] = obj.dataPreProcess(sample{1}, sample{2});
                end
                assert(size(X,2) == 1, "绘制3D结构时，仅支持一个样本。");

                % 图像信息
                plotSingleSample(X, obj.unitSize(1), obj.unitWidth(1), dis(1), "Amp");
                plotSingleSample(Y, obj.unitSize(end), obj.unitWidth(end), dis(end), "Amp");

                colormap jet;

            elseif(nargin > 2)

                % 禁止sample提供Y
                assert(length(sample) == 1, "在修改输出观测层情况下，样本仅支持输入X。");
                X = obj.dataPreProcess(sample{1});
                assert(size(X,2) == 1, "绘制3D结构时，仅支持一个样本。");

                % D2NN网络最后一层散射矩阵重算
                if(nargin == 4)
                    Dnew = obj.layerDistance(end);
                end
                Wlast = obj.WGenerate([obj.unitWidth(end-1), Wnew], [obj.unitSize(end-1), Nnew], Dnew, obj.frequency);

                % 计算样本X输出
                Y = obj.netPredictLayers(X);
                Y = Wlast * Y{end-1};


                % 图像信息
                plotSingleSample(X, obj.unitSize(1), obj.unitWidth(1), dis(1), "Amp");
                plotSingleSample(Y, Nnew, Wnew, dis(end-1) + Dnew , "Amp");

                colormap jet;
            end

            % 结构标注
            inInfo = sprintf("INPUT: %d [%.1fmm]", obj.unitSize(1), obj.unitWidth(1));
            midInfo = sprintf("MID: %s [%.1fmm]", obj.lizardTail(sprintf('%d-', obj.unitSize(2:end-1))), obj.unitWidth(2));
            outInfo = sprintf("OUTPUT: %d [%.1fmm]", obj.unitSize(end), obj.unitWidth(end));
            disInfo = sprintf("DIS: [%smm]", obj.lizardTail(sprintf('%d-', obj.layerDistance)));
            title([inInfo, midInfo, outInfo, disInfo], ...
                "HorizontalAlignment","center",VerticalAlignment="bottom");
            axis equal;

            hold off;

            function plotSingleLayer(u, w, d, rgb)
                %plotSingleLayer （内嵌函数）绘制单层D2NN
                %   Input:
                %   u:单元边长数, w:边长, d:z向距离, rgb:颜色

                l = (u-1)*w/2;

                % 外框
                line([d d d d d], [-l -l l l -l],[l -l -l l l], 'Color', rgb, 'LineWidth', 1.5);
                
                % 内部线
                arrayfun(@(x) line([d d], [-l l], [x x]*w, 'Color', (rgb+1)/2), (2:u-1) - (u+1)/2);
                arrayfun(@(x) line([d d], [x x]*w, [-l l], 'Color', (rgb+1)/2), (2:u-1) - (u+1)/2);
            end

            function plotSingleSample(img, u, w, d, type)
                %plotSingleSample （内嵌函数）绘制单层样本
                %   Input:
                %   img:列向量, u:单元边长数, w:边长, d:z向距离
                %   type: 'Amp'/'Pha',绘制幅度/相位信息
                
                if(nargin == 4)
                    type = "Amp";
                end

                % 归一化操作
                if(strcmpi(type, "Amp"))
                    img = rescale(abs(img));
                elseif(strcmpi(type, "Pha"))
                    img = rescale(angle(img));
                end

                % 绘图
                [X0,Y0] = meshgrid(((1:u)-(u+1)/2)*w,((1:u)-(u+1)/2)*w);
                X0 = X0(:);
                Y0 = Y0(:);
                D = d * ones(size(X0));

                scatter3(D, X0, Y0, 64, img, 'filled', 'Marker','square', ...
                    'MarkerEdgeColor', 'none');


            end
        end

        function plotTrainingRMSECurves(obj)
            %plotTrainingCurves 绘制训练结果曲线

            if(isempty(obj.iter))
                warning("模型未训练");
                return;
            end

            hold on;

            % iter0 = obj.iter - 1;
            iter0 = length(log10(obj.miniBatchRMSE));

            % Minibatch RMSE
            p1 = plot(1:iter0, 10*log10(obj.miniBatchRMSE));

            % Verbose RMSE
            p2 = plot(obj.trainingOptions.VerboseFrequency:obj.trainingOptions.VerboseFrequency:iter0, 10*log10(obj.verboseRMSE), ...
                'Marker', '*');

            % Validation RMSE
            if(obj.isValidationSet)
                p3 = plot(obj.trainingOptions.VerboseFrequency:obj.trainingOptions.VerboseFrequency:iter0, 10*log10(obj.validationRMSE), ...
                    'Marker', '*');
                legend([p1, p2, p3], ["MiniBatch RMSE", "Verbose RMSE", "Validation RMSE"]);

            else
                legend([p1,p2], ["MiniBatch RMSE", "Verbose RMSE"]);
            end


            % mu Alternation Intervals
            batchNum = ceil(obj.sampleNum / obj.trainingOptions.MiniBatchSize);
            ind =  1:obj.trainingOptions.LearnRateDropPeriod*batchNum:length(obj.miniBatchRMSE);
            ind = [ind length(obj.miniBatchRMSE)];
            yl = ylim;
            for ii = 1:length(ind)-1
                color = [0 0 0] + mod(ii+1,2);
                p = patch(ind([ii ii ii+1 ii+1 ii]), yl([1 2 2 1 1]), color, 'EdgeColor', 'none', 'FaceAlpha', 0.05);
                set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
            end


            grid on;
            hold off;

        end

        function plotTrainingAccuCurves(obj)
            %plotTrainingAccuCurves 绘制分类问题中中准确率曲线

            if(obj.netType ~= obj.netTypeClassification)
                warning('D2NN类型非分类网络。图形未绘制。');
                return;
            end

            if(isempty(obj.iter))
                warning("模型未训练");
                return;
            end

            hold on;

            % iter0 = obj.iter - 1;
            iter0 = length(log10(obj.miniBatchAccu));

            % Minibatch RMSE
            p1 = plot(1:iter0, obj.miniBatchAccu);

            % Verbose Accu
            p2 = plot(obj.trainingOptions.VerboseFrequency:obj.trainingOptions.VerboseFrequency:iter0, obj.verboseAccu, ...
                'Marker', '*');

            % Validation Accu
            if(obj.isValidationSet)
                p3 = plot(obj.trainingOptions.VerboseFrequency:obj.trainingOptions.VerboseFrequency:iter0, obj.validationAccu, ...
                    'Marker', '*');
                legend([p1, p2, p3], ["MiniBatch Accu", "Verbose Accu", "Validation Accu"], 'Location','best');

            else
                legend([p1,p2], ["MiniBatch Accu", "Verbose Accu"], 'Location','best');
            end

            grid on;

            % mu Alternation Intervals
            batchNum = ceil(obj.sampleNum / obj.trainingOptions.MiniBatchSize);
            ind =  1:obj.trainingOptions.LearnRateDropPeriod*batchNum:length(obj.miniBatchAccu);
            ind = [ind length(obj.miniBatchAccu)];
            yl = ylim;
            for ii = 1:length(ind)-1
                color = [0 0 0] + mod(ii+1,2);
                p = patch(ind([ii ii ii+1 ii+1 ii]), yl([1 2 2 1 1]), color, 'EdgeColor', 'none', 'FaceAlpha', 0.05);
                set(get(get(p,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
            end
            
            hold off;



        end
        
        function plotPhase(obj)
            %plotPhase 绘制训练结束后相位分布

            % 子图行数
            row = ceil((obj.layerNum-2)/4);
            % 子图列数
            col = min(4, (obj.layerNum-2));
            % 中间N-2层
            for ii = 1:obj.layerNum-2
                subplot(row,col,ii);
                imagesc(reshape(angle(obj.M{ii+1}), obj.unitSize(ii+1), obj.unitSize(ii+1)));
                axis square;
            end

        end



        % = = = = = CST接口 = = = = = %
        % =========================== %

        function CSTFileWrite(obj,filename, pha2strFcn, supInfo)
            %CSTFileWrite CST接口，导出为txt文件
            % Input:
            %   filename        写入txt文件名称
            %   pha2strFcn      相位转换为字符串函数（函数句柄）
            %   supInfo         （允许缺省）补充信息
            %
            % 导出格式：
            %   1. 一个TXT文件，每一层数据段中占一行，第一层、最后一层不导出。
            %   2. 文件内容：
            %       Filename            文件名
            %       Export Date         导出日期
            %       D2NN Info:          D2NN信息
			%			Frequency 		频率
            %           Unit Number     单元数
            %           Unit Width      单元边长
            %           Layer Distance  层间距
            %       Supplementary Info  补充信息
            %
            %       {数据段}
            %       #%d#%d#%f#%s
            %           %d: 当前层，最小为2
            %           %d: 当前层单元数(Number)
            %           %f: 当前层层距离(Distance)
            %           %s: 相位信息, pha2strFcn(angle(obj.M))结果
            %
            %

            if(nargin < 4)
                supInfo = "None";
            end

            % 打开文件
            fid = fopen(filename, "w+");

            % 辅助信息
            fprintf(fid, '\\\\ Filename: %s\n', filename);
            fprintf(fid, '\\\\ Export Date: %s\n', datestr(datetime("now")));

            % D2NN信息
            fprintf(fid, '\\\\ D2NN Info:\n');
            typeEnum = {'Classification', 'Regression'};
            fprintf(fid, '\\\\\tNetwork Type:%d-%s\n', obj.netType, typeEnum{obj.netType+1});
			fprintf(fid, '\\\\\tFrequency: %fHz\n', obj.frequency);
            fprintf(fid, '\\\\\tLayer Number: %d\n', obj.layerNum);
            fprintf(fid, '\\\\\tUnit Number: %d-[%s]-%d\n', obj.unitSize(1), ...
                obj.lizardTail(sprintf('%d-', obj.unitSize(2:end-1))), ...
                obj.unitSize(end));
            fprintf(fid, '\\\\\tUnit Width: %.1f-[%s]-%.1f\n', obj.unitWidth(1), ...
                obj.lizardTail(sprintf('%.1f-', obj.unitWidth(2:end-1))), ...
                obj.unitWidth(end));
            fprintf(fid, '\\\\\tLayer Distance: %s\n', ...
                obj.lizardTail(sprintf('%.1f-', obj.layerDistance)));
            fprintf(fid, '\\\\ Supplementary Info:\n\\\\\t%s\n', supInfo);

            % 数据段
            for iter0 = 2:obj.layerNum-1

                % iter0行数据段
                fprintf(fid, '#%d', iter0);
                fprintf(fid, '#%d', obj.unitSize(iter0));
                fprintf(fid, '#%f', sum(obj.layerDistance(1:iter0-1)));

                fprintf(fid, '#');
                cellfun(@(x) fprintf(fid, '%s ', x), arrayfun(pha2strFcn, angle(obj.M{iter0}), 'UniformOutput', false));
                fprintf(fid, '\n');
            end


            % 关闭文件
            fclose(fid);

        end

    end


    % = = = =  用户友好函数 = = =  = %
    % =========================== %
    methods (Access = public)

        % = = 调制矩阵obj.M对外接口 = = %
        % =========================== %
        function obj = setM(obj, iLayer, inputM)
            %setM 直接对第iLayer层权重矩阵obj.M{iLayer}赋值

            assert(numel(obj.M{iLayer}) == numel(inputM), "数据大小不一致");

            obj.M{iLayer} = inputM;
        end

    end
    
    methods (Access = protected)

        % = = = = = 辅助函数 = = = = = %
        % ============================ %

        function [X,Y] = dataPreProcess(obj, X, Y)
            %dataPreProcess 输入/输出数据预处理
            %   统一输入数据为二维数组

            % 例：X/Y维度应为2*2*1000或4*1000
            % 其中，2为单元边长数，1000为样本数
            assert((size(X,1) == obj.unitSize(1) && ...
                size(X,2) == obj.unitSize(1)) ||...
                size(X,1) == obj.unitNums(1), ...
                sprintf("X样本维度(%s)与网络首层(%d)不一致", ...
                obj.lizardTail(sprintf('%d-,',size(X))), obj.unitNums(1)));

            % 例：统一拉伸为4*1000大小
            X = reshape(X, obj.unitNums(1), []);

            if(nargin == 3)
                assert((size(Y,1) == obj.unitSize(end) && ...
                    size(Y,2) == obj.unitSize(end)) ||...
                    size(Y,1) == obj.unitNums(end), ...
                    sprintf("y样本维度(%s)与网络末层(%d)不一致", ...
                    obj.lizardTail(sprintf('%d-',size(Y))), obj.unitNums(end)));

                Y = reshape(Y, obj.unitNums(end), []);


                assert(size(X,2) == size(Y,2), ...
                    sprintf("X样本数(%d)与Y样本数(%d)不一致", ...
                    size(X,2), size(Y,2)));
            end

        end
                
        function W = WGenerate(obj, width, unitSize, dis, freq)    
            %WGenerate 基于位置分布生成第ii个散射矩阵

            % 角波数
            k = 2 * pi * freq / obj.c0;

            % 子边框
            l1 = width(1) * ((1:unitSize(1)) - (unitSize(1)+1)/2);
            l2 = width(2) * ((1:unitSize(2)) - (unitSize(2)+1)/2);

            [x1, y1] = meshgrid(l1, l1);
            [x2, y2] = meshgrid(l2, l2);

            % 空间坐标
            x1 = x1(:)'; y1 = y1(:)';
            x2 = x2(:); y2 = y2(:);

            r = sqrt((x1-x2).^2 + (y1-y2).^2 + dis^2);

            % 散射矩阵
            W = exp(-1j * k * r) ./ (r.^2);

            % 归一化
            % 矩阵2范数：|W|_2 = sqrt[λ_max[W*W^H]]
            W = W / norm(W, 2);
        end

        function WI = inverseM(obj)
            %inverseM 计算梯度时辅助变量
            WI = cell(obj.layerNum-1, 1);

            WI{end} = obj.W{end}.';

            for kk = (obj.layerNum-2):-1:2
                WI{kk} = obj.W{kk}.' * (obj.M{kk+1} .*  WI{kk+1});
            end
        end

        function Y = netPredictLayers(obj, x)
            %netPredictLayers 输出通过网络逐层结果
            Y = cell(obj.layerNum-1, 1);

            Y{1} = x;

            for kk = 2:obj.layerNum
                Y{kk} = obj.M{kk} .* (obj.W{kk-1} * Y{kk-1});
            end

        end

        % = = = = = 核心函数 = = = = = %
        % ============================ %

        function obj = gradientUpdate(obj, X, Y)
            %gradientUpdate 权重更新

            YL = netPredictLayers(obj, X);

            % 输出与输出模值
            U = YL{end};
            O = U .* conj(U);


            % 第一部分 dL/dO
            switch(obj.netType)
                case(obj.netTypeClassification)
                    
                    % 分类问题, Loss = CrossEntropy(O,Y)
                    O = softmax(O);
                    dlo = O - Y;

                case(obj.netTypeRegression)

                    % 回归问题, Loss = |O-Y|_2^2
                    
                    % 方法A: 输出反归一化
                    O = O ./ vecnorm(O) .* vecnorm(Y); % (暂时凑合对付)
                    
                    dlo = 2 * (O - Y);
            end

            P2 = dlo .* conj(U);
            
            % 第二部分 U{N+1}
            % 就是U

            % 第三部分 dUm/dPhi
            WI = inverseM(obj);

            for kk = 2:(obj.layerNum-1)
                % 1. real[1j * z] = -imag[z]
                % 2. G = G - 2*imag(diag(A(:,ii)) * B * C(:,ii)); {ii = 1:256}
                %    Equals to 
                %    G = -2*sum(imag(A.* (B * C)), 2);

                % 梯度计算
                gradient = -2 * sum(imag(YL{kk} .* (WI{kk} * P2)), 2) / obj.trainingOptions.MiniBatchSize;

                % ADAM算法
                obj.m{kk} = obj.trainingOptions.GradientDecayFactor * obj.m{kk} + (1-obj.trainingOptions.GradientDecayFactor) * gradient;
                obj.v{kk} = obj.trainingOptions.SquaredGradientDecayFactor * obj.v{kk} + (1-obj.trainingOptions.SquaredGradientDecayFactor) * gradient.^2;
                m0 = obj.m{kk} / (1-obj.trainingOptions.GradientDecayFactor^obj.iter);
                v0 = obj.v{kk} / (1-obj.trainingOptions.SquaredGradientDecayFactor^obj.iter);

                % ADAM算法更新
                obj.M{kk} = obj.M{kk} .* exp(-1j * obj.mu * (m0 ./ (sqrt(v0) + obj.trainingOptions.Epsilon)));

            end

            % 计算MinibatchLoss
            
            switch(obj.netType)
                case(obj.netTypeClassification)
                    
                    % 分类问题, Loss = CrossEntropy(O,Y)
                    obj.miniBatchRMSE(obj.iter) = mean(-log(O(find(Y)))); %#ok<FNDSB> %(有待改进)
                    
                    [~,ia] = max(O);
                    obj.miniBatchAccu(obj.iter) = mean(Y(ia + (0:size(Y,1):size(Y,1)*length(ia)-1))); 


                case(obj.netTypeRegression)

                    % 回归问题, Loss = |O-Y|_2^2
                    obj.miniBatchRMSE(obj.iter) = mean(sum((O - Y).^2));

            end
        end


        % = = = = = 辅助函数 = = = = = %
        % ============================ %

        function obj = trainD2NNSimplify(obj, trainX, trainY, trainingOptions, iterMax)
            %trainD2NNSimplify 用于最优学习率确定的简化版训练函数
            % Input:
            %   trainX, trainY      训练集合X,Y
            %   trainingOptions     训练参数
            %   iterMax             最大迭代次数
            %
            % 与非简化版区别
            % 1. 关闭VerboseRMSE计算（全数据集计算开销太大），以miniBatchRMSE为判决标准
            % 2. 无用的功能关闭，如计时，显示，验证集数据
            % 3. iterMax存在，允许一个Epoch不跑完就退出
            % 4. 暂时不支持Mu梯度下降

            % 样本数
            obj.sampleNum = size(trainX, 2);

            % trainOpt复制过来
            obj.trainingOptions = trainingOptions;
            obj.mu = trainingOptions.InitialLearnRate;

            % 学习批次数
            batchNum = ceil(obj.sampleNum / obj.trainingOptions.MiniBatchSize);
            batchInd = (1:obj.trainingOptions.MiniBatchSize)' + ...
                round(linspace(1,obj.sampleNum-obj.trainingOptions.MiniBatchSize, batchNum));

            % 全数据集RMSE，以此为依据
            obj.verboseRMSE = 0;
            obj.miniBatchRMSE = zeros(min(iterMax, obj.trainingOptions.MaxEpochs * batchNum), 1);

            % ADAM动量项初始化
            obj.m = arrayfun(@(x) zeros(x,1), obj.unitNums, 'UniformOutput', 0);
            obj.v = arrayfun(@(x) zeros(x,1), obj.unitNums, 'UniformOutput', 0);

            obj.iter = 1;
            for epoch = 1:obj.trainingOptions.MaxEpochs
                for batch = 1:batchNum
                    obj = obj.gradientUpdate(trainX(:,batchInd(:, batch)), trainY(:,batchInd(:, batch)));
                    obj.iter = obj.iter+1;
                    if(obj.iter > iterMax)
                        break;
                    end
                end
                if(obj.iter > iterMax)
                    break;
                end
            end
            % obj.verboseRMSE = obj.lossCalFcn(trainX, trainY);
        end
    
    end
end

