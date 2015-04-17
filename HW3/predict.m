function p = predict(svmModel, X)
%predict labels for given input X and weight vector w (using LR)
%and return the score (NOT The predicted class)
p = ((X*svmModel.w)+svmModel.w0);