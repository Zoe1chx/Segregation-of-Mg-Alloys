function plotConcentrationRow(C, midIndex, figNumber)
   colors = [0.8, 0.2, 0.2;    
              0, 0, 0;         
              0, 0.3, 0.7;      
              0.1, 0.6, 0.3];   
    lineStyles = {'-', '--', ':', '-.'};

    y = linspace(0, 0.09, size(C{1}, 1));  
    figure(figNumber);  
    hold on;            
    for i = 1:length(C)
        m = C{i}(midIndex, :);  
        middleRow=m/0.06;
        plot(y,middleRow, 'LineWidth', 3, 'Color', colors(mod(i-1, size(colors, 1)) + 1, :), 'LineStyle', lineStyles{mod(i-1, length(lineStyles)) + 1});
    end
      ylim([0.5, 2]);
     xticks(linspace(0, 0.09, 4));
     yticks([ 0,0.5, 1,1.5,2]); 
      xlabel('{\itx} (m)', 'FontSize', 26, 'FontName', 'Arial', 'FontWeight', 'bold');
     ylabel('{\itC/C\infty}', 'FontSize', 26, 'FontName', 'Arial', 'FontWeight', 'bold');
     set(gca, 'Box', 'on', 'LineWidth', 2, 'FontSize', 16,'FontName', 'Arial', 'FontWeight', 'bold', 'GridLineStyle', '--');
      xlim([0,0.09])

     legendStr = arrayfun(@(t) sprintf('\\it t = %ds', t), [1, 5, 10, 15], 'UniformOutput', false);
    legend(legendStr, 'Location', 'best');
    hold off;
end
