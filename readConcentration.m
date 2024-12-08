function C = readConcentration(filename, range)
    [C, ~, ~] = xlsread(filename, range);
end