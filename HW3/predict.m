function p = predict(svmModel, X)
%predict labels for given input X and weight vector w (using LR)
p = ((X*svmModel.w)+svmModel.w0);