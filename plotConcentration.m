function plotConcentration(C, midIndex, figNumber)
      colors = [0.8, 0.2, 0.2;    
              0, 0, 0;          
              0, 0.3, 0.7;    
              0.1, 0.6, 0.3];   
    lineStyles = {'-', '--', ':', '-.'};
    x = linspace(0, 0.3, size(C{1}, 2));
    figure(figNumber);
    hold on;
    % Initialize to find the range for the y-axis
    minY = inf;
    maxY = -inf;
    % Plot each line
    for i = 1:length(C)
        m = C{i}(:,midIndex);  
        middleColumn =m/0.06;
        plot(x, middleColumn, 'LineWidth', 3, 'Color', colors(mod(i-1, size(colors, 1)) + 1, :), 'LineStyle', lineStyles{mod(i-1, length(lineStyles)) + 1});
        % Update y-axis range
        minY = min(minY, min(middleColumn));
        maxY = max(maxY, max(middleColumn));
    end
    % Set y-axis limits to fixed range
    ylim([0.02, 0.1]); % Adjust the range as needed
    % Set x and y labels
    xlabel('{\ity} (m)', 'FontSize', 26, 'FontName', 'Arial', 'FontWeight', 'bold');
    ylabel('{\itC/C\infty}', 'FontSize', 26, 'FontName', 'Arial', 'FontWeight', 'bold');
    set(gca, 'Box', 'on', 'LineWidth', 2, 'FontSize', 16,'FontName', 'Arial', 'FontWeight', 'bold', 'GridLineStyle', '--');
    ylim([0, 2]);
    yticks([ 0,0.5, 1,1.5,2]);  % Fixed y-axis ticks as specified
    % Set x-axis ticks to have exactly five values
    xticks(linspace(0, 0.3, 6));
     xlim([-0.06, 0.3]); % Adjust the range as needed
    % Add a legend
    legendStr = arrayfun(@(t) sprintf('\\it t = %ds', t), [1, 5, 10, 15], 'UniformOutput', false);
    legend(legendStr, 'Location', 'best');
    hold off;
end

