clc;
clear all;
close all;

% Importing the data
data = csvread('drugdata_noname.csv', 1, 0);
X = data(:, 1:5);
y = data(:, 6);
yy = y;

% Data preprocessing
X = zscore(X); % Apply normalization to input features
[y, mu_target, sigma_target] = zscore(y);

% Split data into training (80%) and testing (20%)
rng(42); % For reproducibility
cv1 = cvpartition(size(X, 1), 'HoldOut', 0.2); % 80% Train, 20% Test
idxTrainVal = training(cv1);
idxTest = test(cv1);

X_trainVal = X(idxTrainVal, :);
y_trainVal = y(idxTrainVal, :);
X_test = X(idxTest, :);
y_test = y(idxTest, :);

% Further split TrainVal into 85% Training & 15% Validation
cv2 = cvpartition(sum(idxTrainVal), 'HoldOut', 0.15);
idxTrain = training(cv2);
idxVal = test(cv2);

X_train = X_trainVal(idxTrain, :);
y_train = y_trainVal(idxTrain, :);
X_val = X_trainVal(idxVal, :);
y_val = y_trainVal(idxVal, :);

numMFs_range = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
clusterRadius_range = [0.511, 0.576, 0.6111, 0.654, 0.711, 0.783, 0.811, 0.87765, 0.911, 0.965, 1];

% Initialize arrays to store the results
results = [];
best_mse_val = inf; % Initialize the best validation MSE to infinity
best_model = struct();

for numMFs = numMFs_range
    for clusterRadius = clusterRadius_range
        % Generate FIS using genfis with custom options
        opt = genfisOptions("SubtractiveClustering");
        opt.ClusterInfluenceRange = clusterRadius;

        % Create the FIS
        fis = genfis(X_train, y_train, opt);

        % Train ANFIS model
        epochs = 300;
        anfisModel = anfis([X_train y_train], fis, epochs);

        % Evaluate ANFIS model on training data
        predicted_y_train = evalfis(X_train, anfisModel);
        y_train_denorm = (y_train * sigma_target + mu_target) / 1000000;
        predicted_y_train_denorm = (predicted_y_train * sigma_target + mu_target) / 1000000;
        error_train = y_train_denorm - predicted_y_train_denorm;
        mse_train = mean(error_train.^2);
        mae_train = mean(abs(error_train));

        % Evaluate ANFIS model on validation data
        predicted_y_val = evalfis(X_val, anfisModel);
        y_val_denorm = (y_val * sigma_target + mu_target) / 1000000;
        predicted_y_val_denorm = (predicted_y_val * sigma_target + mu_target) / 1000000;
        error_val = y_val_denorm - predicted_y_val_denorm;
        mse_val = mean(error_val.^2);
        mae_val = mean(abs(error_val));

        % Evaluate ANFIS model on testing data
        predicted_y_test = evalfis(X_test, anfisModel);
        y_test_denorm = (y_test * sigma_target + mu_target) / 1000000;
        predicted_y_test_denorm = (predicted_y_test * sigma_target + mu_target) / 1000000;
        error_test = y_test_denorm - predicted_y_test_denorm;
        mse_test = mean(error_test.^2);
        mae_test = mean(abs(error_test));

        % Calculate R-squared for training data
        SSres_train = sum(error_train.^2);
        SStot_train = sum((y_train_denorm - mean(y_train_denorm)).^2);
        rsquared_train = 1 - SSres_train / SStot_train;

        % Calculate R-squared for validation data
        SSres_val = sum(error_val.^2);
        SStot_val = sum((y_val_denorm - mean(y_val_denorm)).^2);
        rsquared_val = 1 - SSres_val / SStot_val;

        % Calculate R-squared for testing data
        SSres_test = sum(error_test.^2);
        SStot_test = sum((y_test_denorm - mean(y_test_denorm)).^2);
        rsquared_test = 1 - SSres_test / SStot_test;

        % Store the results
        results = [results; numMFs, clusterRadius, mse_train, mse_val, mse_test, mae_test, rsquared_train, rsquared_val, rsquared_test];

        % Check if the current model is the best based on the validation MSE
        if mse_val < best_mse_val
            best_mse_val = mse_val;
            best_model.anfisModel = anfisModel;
            best_model.numMFs = numMFs;
            best_model.clusterRadius = clusterRadius;
            best_model.mse_train = mse_train;
            best_model.mae_train = mae_train;
            best_model.mse_val = mse_val;
            best_model.mse_test = mse_test;
            best_model.mae_test = mae_test;
            best_model.rsquared_train = rsquared_train;
            best_model.rsquared_val = rsquared_val;
            best_model.rsquared_test = rsquared_test;
        end
    end
end

% Save results to a CSV file
results_table = array2table(results, 'VariableNames', {'NumMFs', 'ClusterRadius', 'MSE_Train', 'MSE_Val', 'MSE_Test', 'MAE_Test', 'RSquared_Train', 'RSquared_Val', 'RSquared_Test'});
writetable(results_table, 'anfis_results.csv');

% Display the best result parameters
disp('Best Model Parameters:');
disp(['Number of Membership Functions: ' num2str(best_model.numMFs)]);
disp(['Cluster Radius: ' num2str(best_model.clusterRadius)]);
disp(['Training Mean Squared Error: ' num2str(best_model.mse_train)]);
disp(['Validation Mean Squared Error: ' num2str(best_model.mse_val)]);
disp(['Testing Mean Squared Error: ' num2str(best_model.mse_test)]);
disp(['Training R-Squared: ' num2str(best_model.rsquared_train)]);
disp(['Validation R-Squared: ' num2str(best_model.rsquared_val)]);
disp(['Testing R-Squared: ' num2str(best_model.rsquared_test)]);

% Plot regression plot for the best model
figure;
predicted_y_all = evalfis(X, best_model.anfisModel);
predicted_y_all_denorm = (predicted_y_all * sigma_target + mu_target);
plot(y, predicted_y_all_denorm, 'o');
hold on;
plot([min(y), max(y)], [min(y), max(y)], 'k--'); % Plot diagonal line (y = x)
hold off;
xlabel('Actual Output');
ylabel('Predicted Output');
title('Regression Plot for Best Model');
legend('Predicted vs Actual', 'Diagonal Line', 'Location', 'best');
grid on;
