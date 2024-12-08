baseNumber = '0015';
baseFolder = 'C:\Users\DELL\Desktop\script\datafinal\Quanzhuding';
fileTypes = {'Temperature_', 'Liquid_fraction_', 'Concentration_', 'U_velocity_', 'V_velocity_'};
range = 'A1:CW101';

% Read data from Excel files
for i = 1:length(fileTypes)
    filename = fullfile(baseFolder, [fileTypes{i} baseNumber '.xlsx']);
    [data{i}, txt, raw] = xlsread(filename, range);
end

% Assign data to variables
[T, F, C, u, v] = deal(data{:});

x1 = linspace(0, 0.09, size(T, 2));
y = linspace(0, 0.3, size(T, 1));
[X, Y] = meshgrid(x1, y);

% Determine the center indices
centerY = (size(C, 1) + 1) / 2;
centerX = (size(C, 2) + 1) / 2;

% Determine 25% indices
upperY = round(centerY +0.25 * size(C, 1));
lowerY = round(centerY - 0.25 * size(C, 1));
leftX = round(centerX - 0.3 * size(C, 2));
rightX = round(centerX -0.5 * size(C, 2));

% Extract concentration data along the horizontal and vertical center lines
horizontalCenterLine = T(centerY, :);
horizontalUpper25 = T(upperY, :);
horizontalLower25 = T(lowerY, :);

verticalCenterLine = T(:, centerX);
verticalLeft25 = T(:, leftX);
verticalRight25 = T(:, rightX);

% Plot the concentration distribution along the horizontal center line
figure;
plot(x1, horizontalCenterLine, 'r-', 'LineWidth', 2, 'DisplayName', 'Center line');
hold on;
plot(x1, horizontalUpper25, 'k--', 'LineWidth', 2, 'DisplayName', 'Upper line');
plot(x1, horizontalLower25, 'b-.', 'LineWidth', 2, 'DisplayName', 'Lower line');

legend;
xlabel('{\itx} (m)', 'FontSize', 26, 'FontName', 'Arial', 'FontWeight', 'bold');
ylabel('{\itT} (K)', 'FontSize', 26, 'FontName', 'Arial', 'FontWeight', 'bold'); % T is italicized
xticks(linspace(0, 0.09, 6));
xlim([0 0.09]);
ylim([0,900])
set(gca, 'Box', 'on', 'LineWidth', 2, 'FontSize', 16,'FontName', 'Arial', 'FontWeight', 'bold', 'GridLineStyle', '--');

x2 = linspace(0, 0.3, size(T, 2));

% Plot the concentration distribution along the vertical center line
figure;
plot(x2, verticalCenterLine, 'r-', 'LineWidth', 2, 'DisplayName', 'Center Line');
hold on;
plot(x2, verticalLeft25, 'k--', 'LineWidth', 2, 'DisplayName', '30% Left line');
plot(x2, verticalRight25, 'b-.', 'LineWidth', 2, 'DisplayName', '50% Left line');
xlabel('{\ity} (m)', 'FontSize', 26, 'FontName', 'Arial', 'FontWeight', 'bold');
ylabel('{\itT} (K)', 'FontSize', 26, 'FontName', 'Arial', 'FontWeight', 'bold'); % T is italicized
xticks(linspace(0, 0.3, 6));

set(gca, 'Box', 'on', 'LineWidth', 2, 'FontSize', 16,'FontName', 'Arial', 'FontWeight', 'bold', 'GridLineStyle', '--');
legend;
