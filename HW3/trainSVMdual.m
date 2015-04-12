function svmModel = trainSVMdual(traindata, trainlabels, C)

%returns svmModel, whcih is a struct with 2 fields:
%svmModel.w and svmModel.w0

%we need to use qpac to solve the quadratic program; it's structure is:
%[x,err,lm] = qp(H,f,L,k,A,b,l,u,display);

%therefore, we need to create the vectors and matrices found from the 
%homework:

[m, n] = size(traindata); 

f = -ones(m,1);

H = (trainlabels*trainlabels.').*(traindata*traindata.');

A = trainlabels.';

b = 0;

ub = C*ones(m,1);

lb = zeros(m,1);

tic
z = qpas(H, f, [], [], A, b, lb, ub);
toc
%z = quadprog(H, f, [], [], A, b, lb, ub);

svmModel.alpha = z;


w = traindata.'*(z.*trainlabels);
w0_all = trainlabels - traindata*w;
svmModel.w = w(1:end);
svmModel.w0 = mean(w0_all);

