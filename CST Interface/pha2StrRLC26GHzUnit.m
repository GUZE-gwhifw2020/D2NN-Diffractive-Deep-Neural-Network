function str = pha2StrRLC26GHzUnit(pha)
%PHA2STRRLC26GHZUNIT 26GHzRLC理想元件单元相位转换CST工程字符串函数
%   此处显示详细说明
%   查表
%   相位 / 幅度 / C0 / C1 / L0 / LL1 单位fF,pF 
T = [-157.50	-0.75	72.709	1078.2	281.75	405.80	0	0
    -135	    -0.69	59.530	1500	267.30	288	    0	0
    -112.50	    -0.58	74.900	1412	16.064	164.80	0	0
    -90	        -0.48	13.200	1348	1794	112.50	0	0
    -67.500	    -0.14	29.910	1895	0.38820	0.24520	0	0
    -45	        -0.25	2000	886.10	5000	0	    0	0
    -22.500	    -0.25	1612	346.60	1138	0	    0	0
    0	        -0.20	1361	187.30	733.70	0	    0	0
    22.500	    -0.37	1291	140.40	398	    0	    0	0
    45	        -0.38	1355	99.200	263.60	0	    0	0
    67.500	    -0.30	1115	68.140	168.70	27.400	0	0
    90	        -0.15	1031	46.960	103.50	16.810	0	0
    112.50	    -0.04	2000	28.520	25.970	3.0930	0	0
    135	        -0.09	65.991	10.249	465.45	1087.1	0	0
    157.50	    -0.36	270.30	0.010	0	    4679.1	0	0
    180	        -0.43	209.20	1675	0	    1061	0	0];

% 量化至1~16下标
parti = linspace(-pi, pi, size(T,1)+1);

ind = quantiz(pha, parti(2:end-1), parti(1:end-1));

str = sprintf('%f %f %f %f', T(ind+1, [3 4 5 6]));

end

