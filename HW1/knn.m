function predlabels = knn(traindata, trainlabels, testdata, k, f, D)

% traindata is a ntrain x ndim matrix that represents your training data (each row is one picture and each column is a feature);
% trainlabels is an ntrain x 1 vector holding the labels (integers between 1 and 8) for the training data;
% testdata is a ntest x ndim matrix that represents your testing data;
% k is number of neighbors;
% f is the distance function you will use (e.g f=?sqeuclidean?);
% D is an ntrain x ntest matrix of precomputed pairwise distances (optional in the function input and your inside code should handle the case where it does not exist, e.g. by calling f); 
% predlabels is ntest x 1 vector holding predicted labels for the test instances.

%check the dimensions of both the incoming vectors
train_dims = size(traindata);
test_dims = size(testdata);

%adjust the matrix dimensions to match
if train_dims(1) > test_dims(1)
    test_dims(train_dims) = 0;
elseif train_dims(1) < test_dims(1)
    train_dims(test_dims) = 0;
end

distance_matrix = pdist2(traindata, test_matrix);
distance_matrix = distance_matrix(:,1)

[row,col] = find(distance_matrix == min(distance_matrix(:)));


predlabels = trainlabels(row(1));
