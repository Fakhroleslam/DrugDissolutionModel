clc;
clear all;
close all;

% Set random seed for reproducibility
rng(42); 

% Importing the data
data = csvread('drugdata_noname.csv', 1, 0);

% Separate the input features (mm, mp, t, p, c) and the output target (y6)
X = data(:, 1:5);
y = data(:, 6);

% Data preprocessing 
[X, mu_X, sigma_X] = zscore(X); 
[y, mu_target, sigma_target] = zscore(y);

% Split data into training and testing sets (80% train, 20% test)
cv = cvpartition(size(X, 1), 'Holdout', 0.2);
trainIdx = training(cv); 
testIdx = test(cv); 

% Split the data
X_train = X(trainIdx, :);
y_train = y(trainIdx); 
X_test = X(testIdx, :);
y_test = y(testIdx);

% Prepare the data for MLP
x1_train = X_train(:,1)'; 
x2_train = X_train(:,2)';
x3_train = X_train(:,3)';
x4_train = X_train(:,4)';
x5_train = X_train(:,5)';
X_train_full = {x1_train; x2_train; x3_train; x4_train; x5_train}; 

% Constructing MLP
net = feedforwardnet([6 8]);

% Set number of inputs and connection architecture
net.numinputs = 5;
net.inputConnect = [1 1 1 1 1; 0 0 0 0 0; 0 0 0 0 0];
net.outputConnect = [0 0 1];
net = configure(net, X_train_full);
net.numLayers = 3;

% Set transfer functions
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'purelin';

% Set training parameters
net.trainFcn = 'trainlm'; 
net.trainParam.epochs = 100000;
net.trainParam.goal = 1e-22;
net.trainParam.min_grad = 1e-10;
net.trainParam.max_fail = 10000;
view(net)  % Visualize the network structure

% Train the network with training data
net = train(net, X_train_full, y_train', 'useParallel', 'yes');  % Use parallel computing

% Predict the output for training data
y_train_pred = net(X_train_full);
y_train_pred = cell2mat(y_train_pred);

% Denormalize the predicted and actual values for training data
y_train_pred_denorm = abs(y_train_pred * sigma_target + mu_target)/1000000;
y_train_denorm = abs(y_train * sigma_target + mu_target)/1000000;
y_train_pred_denorm = y_train_pred_denorm';

% Calculate metrics for training data
mae_train = mean(abs(y_train_denorm - y_train_pred_denorm));
mse_train = mean((y_train_denorm - y_train_pred_denorm).^2);
rrmse_train = sqrt(sum((y_train_denorm - y_train_pred_denorm).^2) / sum(y_train_denorm.^2));
R2_train = 1 - sum((y_train_denorm - y_train_pred_denorm).^2) / sum((y_train_denorm - mean(y_train_denorm)).^2);
AAPRE_train = 100 * mean(abs(y_train_denorm - y_train_pred_denorm) ./ y_train_denorm);
RAE_train = (sum(abs(y_train_denorm - y_train_pred_denorm)) / sum(abs(y_train_denorm - mean(y_train_denorm)))) * 100;

% Display the results for training data
disp('-------------------Model Performance Metrics on Training Data:------------------');
disp(['Mean Absolute Error (MAE): ' num2str(mae_train)]);
disp(['Mean Squared Error (MSE): ' num2str(mse_train)]);
disp(['Root Relative Mean Squared Error (RRMSE): ' num2str(rrmse_train)]);
disp(['R²: ' num2str(R2_train)]);
disp(['Relative Absolute Error Percentage (RAE%): ' num2str(RAE_train) '%']);

% Prepare the test data
x1_test = X_test(:,1)';  % First input feature for testing
x2_test = X_test(:,2)';  % Second input feature for testing
x3_test = X_test(:,3)';  % Third input feature for testing
x4_test = X_test(:,4)';  % Fourth input feature for testing
x5_test = X_test(:,5)';  % Fifth input feature for testing
X_test_full = {x1_test; x2_test; x3_test; x4_test; x5_test};  % Combine input features into a cell array for testing

% Predict the output for test data
y_test_pred = net(X_test_full);
y_test_pred = cell2mat(y_test_pred);

% Denormalize the predicted and actual values for testing data
y_test_pred_denorm = abs(y_test_pred * sigma_target + mu_target)/1000000;
y_test_denorm = abs(y_test * sigma_target + mu_target)/1000000;
y_test_pred_denorm = y_test_pred_denorm';


% Calculate metrics for test data
mae_test = mean(abs(y_test_denorm - y_test_pred_denorm));
mse_test = mean((y_test_denorm - y_test_pred_denorm).^2);
rrmse_test = sqrt(sum((y_test_denorm - y_test_pred_denorm).^2) / sum(y_test_denorm.^2));
R2_test = 1 - sum((y_test_denorm - y_test_pred_denorm).^2) / sum((y_test_denorm - mean(y_test_denorm)).^2);
AAPRE_test = 100 * mean(abs(y_test_denorm - y_test_pred_denorm) ./ y_test_denorm);
RAE_test = (sum(abs(y_test_denorm - y_test_pred_denorm)) / sum(abs(y_test_denorm - mean(y_test_denorm)))) * 100;

% Display the results for test data
disp('----------------Model Performance Metrics on Test Data--------------:');
disp(['Mean Absolute Error (MAE): ' num2str(mae_test)]);
disp(['Mean Squared Error (MSE): ' num2str(mse_test)]);
disp(['Root Relative Mean Squared Error (RRMSE): ' num2str(rrmse_test)]);
disp(['R²: ' num2str(R2_test)]);
disp(['Relative Absolute Error Percentage (RAE%): ' num2str(RAE_test) '%']);


% Plot testing data
scatter(y_test_denorm, y_test_pred_denorm, 60, 'b', 'o');  
hold on;

% Plot training data
scatter(y_train_denorm, y_train_pred_denorm, 60, 'r', 'o');

% Plot actual line
plot(y_test_denorm, y_test_denorm, '--k', 'LineWidth', 2);

% Ensure equal axis scaling
axis square;

% Set font size and make it bold
set(gca, 'FontSize', 16, 'FontWeight', 'bold');

% Move the legend to bottom-right or another less intrusive spot
legend('Predicted Test', 'Predicted Train', 'Actual Line', ...
    'FontSize', 16, 'FontWeight', 'bold', 'Location', 'southeast');

% Customize axis labels with bold font
xlabel('Expected value (mole fraction)', 'FontSize', 16, 'FontWeight', 'bold');
ylabel('Predicted value (mole fraction)', 'FontSize', 16, 'FontWeight', 'bold');


% Set figure size to be square
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperPosition', [0 0 8.5 8.5]);

% Save the figure with high resolution
print('my_figure', '-dpng', '-r600');

hold off;


