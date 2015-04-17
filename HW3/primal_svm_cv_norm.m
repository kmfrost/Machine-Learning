%svm script
clear all
load('hw3_parsed.mat');

C = [0.5, 1, 3, 10, 30, 100, 300, 500, 1000];
AUC = zeros(1, length(C));
idx = 1;
norm_train_data = feat_norm(traindata);
norm_val_data = feat_norm(valdata);
for c=C
    fprintf('C = %d\n', c);
    svmModel = trainSVMprimal(norm_train_data, trainlabels, c);
    y_hat = predict(svmModel, norm_val_data);
    [X, Y, T, AUC(idx)] = perfcurve(vallabels, y_hat, '1');
    fprintf('AUC: %d\n', AUC(idx)); 
    
    perc_correct = sum(sign(y_hat) == vallabels)/length(y_hat);
    fprintf('Percentage correct: %d\n\n', perc_correct);
    idx = idx+1;
end

plot([1:length(C)], AUC)
[val, idx] = max(AUC);

fprintf('Use C = %d\n', C(idx));

[m1, n1] = size(testdata);
norm_test_data = feat_norm(testdata);
svmModel = trainSVMprimal(norm_train_data, trainlabels, C(idx));
preds = predict(svmModel, norm_test_data);

csv_data = [(1:m1)' preds];
dlmwrite('SVM_primal_c100.csv', 'EventID,Prediction', 'delimiter', '', 'coffset', 1);
dlmwrite('SVM_primal_c100.csv', csv_data, '-append');
