%svm script
clear all
load('hw3_parsed.mat');

C = [0.5, 1, 3, 10, 30, 100, 300, 500, 1000];
AUC = zeros(1, length(C));
idx = 1;
for c=C
    fprintf('C = %d\n', c);
    svmModel = trainSVMprimal(traindata, trainlabels, c);
    y_hat = predict(svmModel, valdata);
    [X, Y, T, AUC(idx)] = perfcurve(vallabels, y_hat, '1');
    disp(AUC(idx));
    idx = idx+1;
end

plot([1:length(C)], AUC)
[val, idx] = max(AUC);

fprintf('Use C = %d\n', C(idx));

[m1, n1] = size(testdata);
svmModel = trainSVMprimal(traindata, trainlabels, C(idx));
preds = predict(svmModel, testdata);

csv_data = [(1:m1)' preds];
dlmwrite('SVM_primal_c100.csv', 'EventID,Prediction', 'delimiter', '', 'coffset', 1);
dlmwrite('SVM_primal_c100.csv', csv_data, '-append');
