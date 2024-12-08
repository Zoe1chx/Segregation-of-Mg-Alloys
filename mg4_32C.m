
basePath1 = 'C:\Users\DELL\Desktop\script\datafinal\Quanzhuding';
filenames = {'Concentration_0001.xlsx', 'Concentration_0005.xlsx', 'Concentration_0010.xlsx', 'Concentration_0015.xlsx'};
range = 'A1:CW101';

C = cell(1, 20);
for i = 1:4
    C{i} = readConcentration(fullfile(basePath1, filenames{i}), range);
 
end

% 计算中间索引
midIndex = (size(C{1}, 1) + 1) / 2;
% 函数绘制图像
plotConcentration(C(1:4), midIndex, 1);

