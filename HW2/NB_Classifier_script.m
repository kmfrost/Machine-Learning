%NB Classifier
clear all
load('SenatorVoting.mat')

[m, n] = size(TrainData);

%find the counts and probabilities of y=1 and y=0
c_y1 = sum(TrainLabel);
p_y1 = c_y1/m;
p_y0 = 1-p_y1;
c_y0 = m-c_y1;
prior = [p_y0 p_y1];

%separate the data by class
data_Y1_indices = find(TrainLabel);
data_Y1 = TrainData(data_Y1_indices, :);
num_Y1 = length(data_Y1_indices);
data_Y0_indices = setdiff([1:m], data_Y1_indices);
data_Y0 = TrainData(data_Y0_indices, :);
num_Y0 = length(data_Y0_indices);

%make a table for each vote, exclude zero votes
%    \
%     \
%      \     Y0           Y1
%       \________________________
%        |
% X = -1 |
%        |
%  X = 1 |
%        |
  
tables = zeros(n, 2, 2);
for i=1:n
    Y0_col = data_Y0(:, i);
    Y0_len = length(find(Y0_col));
    Y1_col = data_Y1(:, i);
    Y1_len = length(find(Y1_col));
    tables(i, 1, 1) = length(find(Y0_col < 0))/Y0_len;
    if tables(i, 1, 1) == 0  % smooth
        tables(i, 1, 1) = 1/Y0_len;
    end
    tables(i, 1, 2) = length(find(Y1_col < 0))/Y1_len;
    if tables(i, 1, 2) == 0
        tables(i, 1, 2) = 1/Y1_len;
    end
    tables(i, 2, 1) = length(find(Y0_col > 0))/Y0_len;
    if tables(i, 2, 1) == 0
        tables(i, 2, 1) = 1/Y0_len;
    end
    tables(i, 2, 2) = length(find(Y1_col > 0))/Y1_len;
    if tables(i, 2, 2) == 0
        tables(i, 2, 2) = 1/Y1_len;
    end
end

[m1, n1] = size(TestData);
predlabels = zeros(m1, 1);
for i = 1:m1
   p0 = prior(1);
   p1 = prior(2);
   for j = 1:n1
       vote = TestData(i, j);
       if vote == -1
           p0 = p0 * tables(j, 1, 1);
           p1 = p1 * tables(j, 1, 2);
       elseif vote == 1
           p0 = p0 * tables(j, 2, 1);
           p1 = p1 * tables(j, 2, 2);
       end
   if p0 > p1
       predlables(i) = 0;
   else
       predlabels(i) = 1;
   end
   end
       
end


csv_data = [(1:m1)', predlabels];
dlmwrite('NB_preds.csv','Senator_ID,Party', 'delimiter', '', 'coffset', 1);
dlmwrite('NB_preds.csv', csv_data, '-append');

