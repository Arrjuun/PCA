clear; close all; clc;

data = load('mnist_train.mat');
X = data.train_X;
numPCA = 2;

[m,n] = size(X);

% Get the mean for each feature on the data
xMean = mean(X,1);

% Subtract Mean for each row
% for i = 1 : m
%     X(i,:) = X(i,:) - xMean;
% end

%Calculate Covariance matrix
covX = cov(X);

%Get the eigen values and the eigen vectors
[V, D] = eig(covX); 
D = diag(D);
% Sort
sortedD = sortrows(D,-1);

%For number or dimensions required, calculate projections
for i = 1 : numPCA 
    projection(:,i) = V(:,D == sortedD(i));
end

newX = X * projection; %Data with reduced dimensions

ind1 = data.train_labels(:,1) == 1;
ind2 = data.train_labels(:,1) == 2;
ind3 = data.train_labels(:,1) == 3;
ind4 = data.train_labels(:,1) == 4;
ind5 = data.train_labels(:,1) == 5;
ind6 = data.train_labels(:,1) == 6;
ind7 = data.train_labels(:,1) == 7;
ind8 = data.train_labels(:,1) == 8;
ind9 = data.train_labels(:,1) == 9;
ind10 = data.train_labels(:,1) == 10;

figure;
plot(newX(ind1,1),newX(ind1,2),'.','Color',[0.00  0.00  1.00]);
set(gca, 'ColorOrder', hot);
hold all
plot(newX(ind2,1),newX(ind2,2),'.','Color',[0.00  0.50  0.00]);
plot(newX(ind3,1),newX(ind3,2),'.','Color',[1.00  0.00  0.00]);
plot(newX(ind4,1),newX(ind4,2),'.','Color',[0.00  0.75  0.75]);
plot(newX(ind5,1),newX(ind5,2),'.','Color',[0.75  0.00  0.75]);
plot(newX(ind6,1),newX(ind6,2),'.','Color',[0.75  0.75  0.00]);
plot(newX(ind7,1),newX(ind7,2),'.','Color',[0.25  0.25  0.25]);
plot(newX(ind8,1),newX(ind8,2),'.','Color',[0.75  0.25  0.25]);
plot(newX(ind9,1),newX(ind9,2),'.','Color',[0.95  0.95  0.00 ]);
plot(newX(ind10,1),newX(ind10,2),'.','Color',[1.00  0.10  0.60]);
hold off;

% figure;
% plot(sortedD);
% 
% percentageVariance = sum(sortedD(1:2,:))/sum(sortedD);
% disp(percentageVariance);