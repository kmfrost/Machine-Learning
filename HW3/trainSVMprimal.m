function svmModel = trainSVMprimal(traindata, trainlabels, C)

%returns svmModel, whcih is a struct with 2 fields:
%svmModel.w and svmModel.w0

%we need to use qpac to solve the quadratic program; it's structure is:
%[x,err,lm] = qp(H,f,L,k,A,b,l,u,display);

%therefore, we need to create the vectors and matrices found from the 
%homework:

[m, n] = size(traindata); 
k = m+n+1;

H = zeros(k);
H(1:n, 1:n) = eye(n);

f = zeros(k,1);
f(n+1:end-1)=C;

y_rep = repmat(trainlabels, 1, n);
A = [-y_rep.*traindata -eye(m) -trainlabels];

b = -1*ones(m,1);

lb = [ones(1,n)*-inf zeros(1,m) -inf].';

tic
z = quadprog(H, f, A, b, [], [], lb, []);
toc

svmModel.w0 = z(end);
svmModel.w = z(1:n);