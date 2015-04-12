%svm script
clear all
load('hw3_parsed.mat');


C = [1, 10, 100, 300, 500];
num_sv = zeros(1, length(C));
%C = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80];
AUC = zeros(1, length(C));
idx = 1;
for c=C
    fprintf('C = %d\n', c);
    svmModel = trainSVMdual(x, y, c);
    num_sv(idx) = nnz(svmModel.alpha);
    y_hat = predict(svmModel, valdata);
    [X, Y, T, AUC(idx)] = perfcurve(vallabels, y_hat, '1');
    disp(AUC(idx));
    idx = idx+1;
end

plot([1:length(C)], AUC)
[val, idx] = max(AUC);

fprintf('Use C = %d', C(idx));

[m1, n1] = size(testdata);
svmModel = trainSVMdual(traindata, trainlabels, C(idx));
preds = predict(svmModel, testdata);

csv_data = [(1:m1)' preds];
dlmwrite('SVM_primal_c100.csv', 'EventID,Prediction', 'delimiter', '', 'coffset', 1);
dlmwrite('SVM_primal_c100.csv', csv_data, '-append');
