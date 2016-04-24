%% Skeleton

clear all;
clc;
close all hidden;


BASE_DIR = '/home/peter/MachineLearning/dati/';

%% List of all directories with train data
TRAIN_DATA_LOCATION = {'Query R/R1','Query R/R2','Query R/R5','Query R/R4'};
% TRAIN_DATA_LOCATION = {'Core/60', 'Core/80', 'Core/100', 'Core/120', 'Core/72'};
%TRAIN_DATA_LOCATION = {'Core/60'};

%% List of all directories with test data (leave {} if test data equals train data)
TEST_DATA_LOCATION = {'Query R/R3'};
%TEST_DATA_LOCATION = {};

%% CHANGE THESE IF TEST == TRAIN
TRAIN_FRAC_WO_TEST = 0.6;
TEST_FRAC_WO_TEST = 0.2;

%% CHANGE THESE IF TEST != TRAIN
TRAIN_FRAC_W_TEST = 0.7;


NON_LINEAR_FEATURES = false;
NORMALIZE_FEATURE = true;
CLEAR_OUTLIERS = true;


%% FEATURE DESCRIPTION:
% 1 -> N map
% 2 -> N reduce
% 3 -> Map time avg
% 4 -> Map time max
% 5 -> Reduce time avg
% 6 -> Reduce time max
% 7 -> Shuffle time avg
% 8 -> Shuffle time max
% 9 -> Bandwidth avg
% 10 -> Bandwidth max
% 11 -> N Users
% 12 -> Data size
% 13 -> N Core
CHOOSE_FEATURES = true;

FEATURES = [1:10, 12, 13]; 			% All the features except Users
% FEATURES = [1, 2, 3, 4, 5]; 
FEATURES_DESCRIPTIONS = {			% These will be used to describe the plot axis
	'N map',
	'N reduce',
	'Map time avg',
	'Map time max',
	'Reduce time avg',
	'Reduce time max',
	'Shuffle time avg',
	'Shuffle time max',
	'Bandwidth avg',
	'Bandwidth max',
	'N Users',
	'Data size',
	'N core'
};



rand('seed', 24);
SHUFFLE_DATA = true;

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);


% SVR_DESCRIPTIONS = {
% 	'Linear SVR'
% };





% --------------------------------------------------------------------------------------------------------
% |														 DO NOT MODIFY 									 |
% |														 UNDER THIS BOX 								 |
% --------------------------------------------------------------------------------------------------------

%% Retrieve the data

train_data = get_all_data_from_dirs(BASE_DIR, TRAIN_DATA_LOCATION);

if CHOOSE_FEATURES
	train_data = [train_data(:, 1) , train_data(:, 2:end)(:, FEATURES)];
	FEATURES_DESCRIPTIONS = FEATURES_DESCRIPTIONS(FEATURES);
end

test_data = [];
if not (isempty(TEST_DATA_LOCATION))
	test_data = get_all_data_from_dirs(BASE_DIR, TEST_DATA_LOCATION);
	if CHOOSE_FEATURES
		test_data = [test_data(:, 1) , test_data(:, 2:end)(:, FEATURES)];
	end
end



M = size(train_data, 2) - 1;   		%% Number of features
N_train = size(train_data, 1);		%% Number of train tuples
N_test = size(test_data, 1);		%% Number of test tuples

complete_data = [train_data ; test_data];


if CLEAR_OUTLIERS
	[clean, indices] = clear_outliers(complete_data);

	train_data = clean(indices <= N_train, :);
	test_data = clean(indices > N_train, :);

	N_train = size(train_data, 1);		%% Number of train tuples
	N_test = size(test_data, 1);		%% Number of test tuples

	complete_data = [train_data ; test_data];
end


if NON_LINEAR_FEATURES

	%% @TODO Add non linear features

end


mu = zeros(M+1, 1);
sigma = ones(M+1, 1);

if NORMALIZE_FEATURE
	[scaled, mu, sigma] = zscore(complete_data);

	train_data = scaled(1:N_train, :);
	test_data = scaled(N_train+1:end, :);

end




if SHUFFLE_DATA
	r = randperm(N_train);
	train_data = train_data(r, :);

	%% There is no need to shuffle test data
end


%% SPLIT THE DATA

cv_data = [];
N_cv = 0;

if isempty(TEST_DATA_LOCATION)
	[train_data, test_data, cv_data] = split_sample(train_data, TRAIN_FRAC_WO_TEST, TEST_FRAC_WO_TEST);
	N_train = size(train_data, 1);
	N_cv = size(cv_data, 1);
	N_test = size(test_data, 1);
else
	[train_data, cv_data, ~] = split_sample(train_data, TRAIN_FRAC_W_TEST, 1-TRAIN_FRAC_W_TEST);
	N_train = size(train_data, 1);
	N_cv = size(cv_data, 1);
end





%% Organize data


y_tr = train_data(:, 1);
X_tr = train_data(:, 2:end);

y_cv = cv_data(:, 1);
X_cv = cv_data(:, 2:end);

y_test = test_data(:, 1);
X_test = test_data(:, 2:end);


mu_y = mu(1);
mu_X = mu(2:end);

sigma_y = sigma(1);
sigma_X = sigma(2:end);


%% DECLARE USEFUL VARIABLES

RMSEs = [];
Cs = [];     
Es = [];
predictions = [];
coefficients = {};
SVs = {};
b = {};





%% SVR

% Parametri per svmtrain
% -s --> tipo di SVM (3 = epsilon-SVR)
% -t --> tipo di kernel (0 = lineare, 1 = polinomiale, 2 = gaussiano, 3 = sigmoide)
% -q --> No output
% -h --> (0 = No shrink)
% -p --> epsilon
% -c --> cost

SVR_DESCRIPTIONS = {};

% fprintf('Training white box model with linear SVR');
% fflush(stdout);
% %% White box model, nCores  LINEAR
% SVR_DESCRIPTIONS{end + 1} = 'Linear SVR'
% [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 0 -q -h 0', C_range, E_range);
% options = ['-s 3 -t 0 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
% model = svmtrain (y_tr, X_tr, options);
% [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model);
% Cs(end + 1) = C;
% Es(end + 1) = eps;
% RMSEs(end + 1) = sqrt (accuracy(2));
% coefficients{end + 1} = model.sv_coef;
% SVs{end + 1} = model.SVs;
% b{end + 1} = - model.rho;
% SVR_DESCRIPTIONS{end + 1} = 'Linear SVR'


% fprintf('Training black box model with polynomial SVR');
% fflush(stdout);
% %% Black box model, Polynomial
% SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR'
% [C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 1 -q -h 0', C_range, E_range);
% options = ['-s 3 -t 1 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
% model = svmtrain (y_tr, X_tr, options);
% [predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model);
% Cs(end + 1) = C;
% Es(end + 1) = eps;
% RMSEs(end + 1) = sqrt (accuracy(2));
% coefficients{end + 1} = model.sv_coef;
% SVs{end + 1} = model.SVs;
% b{end + 1} = - model.rho;


fprintf('\n\nTraining black box model with RBF SVR\n');
fflush(stdout);
%% Black box model, RBF (Radial Basis Function)
SVR_DESCRIPTIONS{end + 1} = 'Radial Basis Function SVR';
[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 2 -q -h 0', C_range, E_range);
options = ['-s 3 -t 2 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
model = svmtrain (y_tr, X_tr, options);

[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet
fprintf('\n Testing results:\n');
fprintf('   RMSE = %f\n', sqrt(accuracy(2)));
fprintf('   R^2 = %f\n', accuracy(3));

Cs(end + 1) = C;
Es(end + 1) = eps;
RMSEs(end + 1) = sqrt (accuracy(2));
coefficients{end + 1} = model.sv_coef;
SVs{end + 1} = model.SVs;
b{end + 1} = - model.rho;


%% LINEAR REGRESSION

X_tr = [ones(N_train, 1) , X_tr]; %% Add the intercept


[theta, ~, ~, ~, results] = regress(y_tr, X_tr);

fprintf('\n\nLinear regression:\n');
fprintf('\n theta: \n'); disp(theta');
fprintf('\n Training results: \n');
fprintf('   R^2 = %f\n', results(1));
fprintf('   F = %f\n', results(2));
fprintf('   p-value = %f\n', results(3));
fprintf('   MSE = %f\n\n', results(4));

predictions(:, end+1) = [ones(N_test, 1) X_test] * theta;

% fprintf('predictions  real values\n');
% disp([predictions(1:10,1) y_test(1:10)]);

y_mean = mean(y_test);
sumresidual = 0;
sumtotal = 0;
for i = 1:N_test
	sumresidual += (y_test(i) - predictions(i, end))^2;
	sumtotal += (y_test(i) - y_mean)^2;
end

linRMSE = sqrt(sumresidual/ (N_test - 1));	%% Root Mean Squared Error
linR2 = 1 - (sumresidual / sumtotal);		%% R^2

fprintf(' Testing results:\n');
fprintf('   RMSE = %f\n', linRMSE);
fprintf('   R^2 = %f\n\n', linR2);
RMSEs(end + 1) = linRMSE; 

b{end+1} = 0;
coefficients{end + 1} = 0;
SVs{end + 1} = 0;
Cs(end + 1) = 0;
Es(end + 1) = 0;

% Removes the intercept
X_tr = X_tr(:, 2:end);


%% PLOTTING SVR vs LR

for svr_index = [1]

	for col = 1:M
		figure;
		hold on;

		% Scatters training and test data
		scatter(X_tr(:, col), y_tr, 'r', 'x');
		scatter(X_test(:, col), y_test, 'b');

		% w = SVs{svr_index}' * coefficients{svr_index};
		x = linspace(min(X_test(:, col)), max(X_test(:, col)));
		xsvr = zeros(length(x), M);
		xsvr(:, col) = x;
		[ysvr, ~, ~] = svmpredict(zeros(length(x), 1), xsvr, model, '-q');	%% quiet

		% plot(x, w(col)*x + b{svr_index}, 'g');
		plot(x, ysvr, 'g', 'linewidth', 1);
		plot(x, x*theta(col+1), 'm', 'linewidth', 1);
		
		legend('Training set', 'Testing set', SVR_DESCRIPTIONS{svr_index}, 'Linear regression', 'location', 'southeast');

		% Display SVR margins
		% plot(x, w(col)*x + b{svr_index} + Es(svr_index), 'k');
		% plot(x, w(col)*x + b{svr_index} - Es(svr_index), 'k');

		% Labels the axes
		xlabel(FEATURES_DESCRIPTIONS(col));
		ylabel('Time');
		title(cstrcat('Linear regression vs ', SVR_DESCRIPTIONS{svr_index})); 
		
		hold off;
		% pause;
	end
end





