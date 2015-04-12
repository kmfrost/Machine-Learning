%svm script
clear all
load('hw3_parsed.mat');

svmModel = trainSVMprimal(traindata, trainlabels, 1);

y_hat = valdata*svmModel.w+svmModel.w0;
good_preds = y_hat == vallabels;
perc_correct = sum(good_preds)/length(vallabels);