function predlabels = NB_Classifier(TrainData, TrainLabel, TestData);
%return prediction labels for a NB Classifier given Training Data, Traning
%Lables, and Test Data

%NB Classifier

[m, n] = size(TrainData);
smooth = 1;

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
for ii=1:n
    Y0_col = data_Y0(:, ii);
    Y0_len = length(find(Y0_col));
    Y1_col = data_Y1(:, ii);
    Y1_len = length(find(Y1_col));
    
    %fill the tables
    %add 'smooth' to each bin for smoothing
    tables(ii, 1, 1) = (length(find(Y0_col < 0))+smooth)/(Y0_len+smooth);
    tables(ii, 1, 2) = (length(find(Y1_col < 0))+smooth)/(Y1_len+smooth);
    tables(ii, 2, 1) = (length(find(Y0_col > 0))+smooth)/(Y0_len+smooth);
    tables(ii, 2, 2) = (length(find(Y1_col > 0))+smooth)/(Y1_len+smooth);
end

[m1, n1] = size(TestData);
predlabels = zeros(m1, 1);
for ii = 1:m1
   p0 = prior(1);
   p1 = prior(2);
   for jj = 1:n1
       vote = TestData(ii, jj);
       if vote == -1
           p0 = p0 * tables(jj, 1, 1);
           p1 = p1 * tables(jj, 1, 2);
       elseif vote == 1
           p0 = p0 * tables(jj, 2, 1);
           p1 = p1 * tables(jj, 2, 2);
       end
   if p0 > p1
       predlables(ii) = 0;
   else
       predlabels(ii) = 1;
   end
   end
       
end