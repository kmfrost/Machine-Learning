%svm script
clear all
load('hw3_parsed.mat');

n_test = 7000;
x=traindata(1:n_test,:);
y=trainlabels(1:n_test);
C = [1, 10, 100, 300, 500];
%C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80];
perc_correct = zeros(1, length(C));
idx = 1;
for c=C
    svmModel = trainSVMdual(x, y, c);
    
    y_hat = predict(svmModel, valdata);
    good_preds = y_hat == vallabels;
    perc_correct(idx) = sum(good_preds)/length(vallabels);
    disp(perc_correct(idx));
    idx = idx+1;
end

plot([1:length(C)], perc_correct)
[val, idx] = max(perc_correct);

disp('Use C = ');
disp(C(idx));