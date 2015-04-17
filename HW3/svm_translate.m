data = [1 0; 2.2 1.6; 3.8 1.6; 5 0; 0.9 1.8; 2.4 4; 4 4.1; 5.1 2.4];
labels = [1; 1; 1; 1; -1; -1; -1; -1];

model = fitcsvm(data, labels);

svmtrain(data, labels, 'showplot', true, 'kernel_function', 'quadratic', 'BoxConstraint', 2, 'method', 'QP')

data2 = data + [zeros(8,1), 10*ones(8,1)];
figure;
svmtrain(data2, labels, 'showplot', true, 'kernel_function','quadratic', 'BoxConstraint', 2, 'method', 'QP')
