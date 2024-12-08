baseNumber = '0015';
baseFolder = 'C:\Users\DELL\Desktop\script\datafinal\Quanzhuding\test';
fileTypes = {'Temperature_', 'Liquid_fraction_', 'Concentration_', 'U_velocity_', 'V_velocity_'};
range = 'A1:CW101';

% Read data from Excel files
for i = 1:length(fileTypes)
    filename = fullfile(baseFolder, [fileTypes{i} baseNumber '.xlsx']);
    [data{i}, txt, raw] = xlsread(filename, range);
end

% Assign data to variables
[T, F, C, u, v] = deal(data{:});
C = (data{3}) / 0.06;

x = linspace(0, 0.045*2, size(T, 2));
y = linspace(0, 0.3, size(T, 1));
[X, Y] = meshgrid(x, y);

% Temperature contour plot
figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 8 6]);
contourf(X, Y, T, 100, 'LineColor', 'none');

colormap(jet); 
colorbar;
caxis([200 1100]); 
hold on;
% Randomly select arrow positions
quiver(X(1:4:end, 1:4:end), Y(1:4:end, 1:4:end), u(1:4:end, 1:4:end), v(1:4:end, 1:4:end), 2, 'k');
xlabel('{\itx} (m)', 'FontSize', 18, 'FontName', 'Arial');
ylabel('{\ity} (m)', 'FontSize', 18, 'FontName', 'Arial');
set(gca, 'Box', 'on', 'LineWidth', 3, 'FontSize', 18, 'FontName', 'Arial');
axis equal;
% Control x and y axis ticks
xticks(linspace(min(x), max(x), 2));
yticks(linspace(min(y), max(y), 5));
ylim([0 0.3]);  % Adjust this line based on your actual data range
xlim([0 0.09]);
% Liquid Fraction contour plot
figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 8 6]);
contourf(X, Y, F, 100, 'LineColor', 'none');
colormap(hot);
colorbar;
xlabel('{\itx} (m)', 'FontSize', 18, 'FontName', 'Arial');
ylabel('{\ity} (m)', 'FontSize', 18, 'FontName', 'Arial');
set(gca, 'Box', 'on', 'LineWidth', 3, 'FontSize', 18, 'FontName', 'Arial');
axis equal;
% Control x and y axis ticks
xticks(linspace(min(x), max(x), 2));
yticks(linspace(min(y), max(y), 5));
ylim([0 0.3]);  % Adjust this line based on your actual data range
xlim([0 0.09]);
% Concentration and velocity field with streamlines and arrows
figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 8 6]);
contourf(X, Y, C, 100, 'LineColor', 'none');
colormap(jet);
hold on;
% Randomly select arrow positions
numArrows = 100; 
randomIdx = randperm(numel(X), numArrows);
%quiver(X(randomIdx), Y(randomIdx), u(randomIdx), v(randomIdx), 1.5, 'k');
xlabel('{\itx} (m)', 'FontSize', 18, 'FontName', 'Arial');
ylabel('{\ity} (m)', 'FontSize', 18, 'FontName', 'Arial');
set(gca, 'Box', 'on', 'LineWidth', 3, 'FontSize', 18, 'FontName', 'Arial');
axis equal;
axis tight;
% Control x and y axis ticks
xticks(linspace(min(x), max(x), 2));
yticks(linspace(min(y), max(y), 5));
ylim([0 0.3]);  % Adjust this line based on your actual data range
xlim([0 0.09]);
caxis([0.4 1.4])
colorbar
% Save the figure
saveas(gcf, 'Field_Plots.png', 'png');
