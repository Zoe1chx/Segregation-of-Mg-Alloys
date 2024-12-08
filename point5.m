time = 1:15; % Time from 1s to 20s
centerConcentration = zeros(1, 15); 
leftConcentration = zeros(1, 15); 
rightConcentration = zeros(1, 15); 
upConcentration = zeros(1, 15); 
downConcentration = zeros(1, 15); 

% Loop through each Excel file
for i = 1:15
  
    filename = sprintf('Concentration_%04d.xlsx', i);
    
   
    data = xlsread(filename);
    
    
    gridSize = size(data);
    centerRowIndex = ceil(gridSize(1) / 2);
    centerColIndex = ceil(gridSize(2) / 2);
    
   
    centerConcentration(i) = data(centerRowIndex, centerColIndex);
    
    
    leftConcentration(i) = data(centerRowIndex, max(1, centerColIndex - 20));
    rightConcentration(i) = data(centerRowIndex, min(gridSize(2), centerColIndex + 20));
    upConcentration(i) = data(max(1, centerRowIndex +20), centerColIndex);
    downConcentration(i) = data(min(gridSize(1), centerRowIndex -20), centerColIndex);
end

% Plot the concentration versus time for the center point and surrounding points
figure;
hold on;
plot(time, centerConcentration/0.06, 'b-o', 'LineWidth', 2, 'DisplayName', 'Center');
plot(time, leftConcentration/0.06, 'r-*', 'LineWidth', 2, 'DisplayName', 'Left');
plot(time, rightConcentration/0.06, 'g-.', 'LineWidth', 2, 'DisplayName', 'Right');
plot(time, upConcentration/0.06, 'm:', 'LineWidth', 2, 'DisplayName', 'Up');
plot(time, downConcentration/0.06, 'k-', 'LineWidth', 2, 'DisplayName', 'Down');
xlabel('{\itt} (s)', 'FontSize', 26, 'FontName', 'Arial', 'FontWeight', 'bold');
ylabel('{\itC/C\infty} ', 'FontSize', 26, 'FontName', 'Arial', 'FontWeight', 'bold');
legend('show', 'Location', 'best',  'LineWidth', 2,'FontName', 'Arial','FontWeight', 'bold', 'FontSize',14);
set(gca, 'Box', 'on', 'LineWidth', 2, 'FontSize', 16, 'FontName', 'Arial', 'FontWeight', 'bold','GridLineStyle', '--');
ylim([0,2]); 
 yticks([0, 0.5,1,1.5,2]); 
 xlim([0,20])
hold off;
