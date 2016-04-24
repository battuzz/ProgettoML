%% Skeleton

%%% TODO
%% - Rescaling features to have meaningful plots
%% - Automatic feature selection (with penalties?)


clear all;
clc;
close all hidden;

BASE_DIR = '/home/peter/MachineLearning/dati/';

%% List of all directories with train data
TRAIN_DATA_LOCATION = {'Query R/R1/Core/60','Query R/R1/Core/72','Query R/R1/Core/90','Query R/R1/Core/100','Query R/R1/Core/120'};
% TRAIN_DATA_LOCATION = {'Core/60', 'Core/80', 'Core/100', 'Core/120', 'Core/72'};
%TRAIN_DATA_LOCATION = {'Core/60'};

%% List of all directories with test data (leave {} if test data equals train data)
TEST_DATA_LOCATION = {'Query R/R1/Core/80'};
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
% FEATURES = [1, 2, 7, 12, 13];

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

%% Choose which SVR models to use
% 1 -> Linear SVR
% 2 -> Polynomial SVR
% 3 -> RBF SVR
MODELS_CHOSEN = [1, 2, 3];
COLORS = {'g', 'b', 'c'};

LINEAR_REGRESSION = false;

rand('seed', 24);
SHUFFLE_DATA = true;

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);



% --------------------------------------------------------------------------------------------------
% |									       DO NOT  MODIFY 								           |
% |										   UNDER THIS BOX 								           |
% --------------------------------------------------------------------------------------------------

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
SVR_DESCRIPTIONS = {};
models = {};

%% SVR

% Parametri per svmtrain
% -s --> tipo di SVM (3 = epsilon-SVR)
% -t --> tipo di kernel (0 = lineare, 1 = polinomiale, 2 = gaussiano, 3 = sigmoide)
% -q --> No output
% -h --> (0 = No shrink)
% -p --> epsilon
% -c --> cost


%% White box model, nCores  LINEAR
if ismember(1, MODELS_CHOSEN)
	fprintf('\nTraining model with linear SVR\n');
	fflush(stdout);
	SVR_DESCRIPTIONS{end + 1} = 'Linear SVR';

	[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 0 -q -h 0', C_range, E_range);
	options = ['-s 3 -t 0 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
	model = svmtrain (y_tr, X_tr, options);

	[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet
	sum_abs = 0;
	sum_rel = 0;
	for i = 1:N_test
		sum_abs += abs(y_test(i) - predictions(i, end));
		sum_rel += abs(y_test(i) - predictions(i, end) / predictions(i, end));
	end
	mean_abs = sum_abs / N_test;
	mean_rel = sum_rel / N_test;
	fprintf('\n Testing results:\n');
	fprintf('   RMSE = %f\n', sqrt(accuracy(2)));
	fprintf('   R^2 = %f\n', accuracy(3));
	fprintf('   Mean abs error = %f\n', mean_abs);
	fprintf('   Mean rel error = %f\n\n', mean_rel);

	models{end + 1} = model;
	Cs(end + 1) = C;
	Es(end + 1) = eps;
	RMSEs(end + 1) = sqrt (accuracy(2));
	coefficients{end + 1} = model.sv_coef;
	SVs{end + 1} = model.SVs;
	b{end + 1} = - model.rho;
end


%% Black box model, Polynomial
if ismember(2, MODELS_CHOSEN)
	fprintf('\nTraining model with polynomial SVR\n');
	fflush(stdout);
	SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR';

	[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 1 -q -h 0', C_range, E_range);
	options = ['-s 3 -t 1 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
	model = svmtrain (y_tr, X_tr, options);

	[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet
	sum_abs = 0;
	sum_rel = 0;
	for i = 1:N_test
		sum_abs += abs(y_test(i) - predictions(i, end));
		sum_rel += abs(y_test(i) - predictions(i, end) / predictions(i, end));
	end
	mean_abs = sum_abs / N_test;
	mean_rel = sum_rel / N_test;
	fprintf('\n Testing results:\n');
	fprintf('   RMSE = %f\n', sqrt(accuracy(2)));
	fprintf('   R^2 = %f\n', accuracy(3));
	fprintf('   Mean abs error = %f\n', mean_abs);
	fprintf('   Mean rel error = %f\n\n', mean_rel);

	models{end + 1} = model;
	Cs(end + 1) = C;
	Es(end + 1) = eps;
	RMSEs(end + 1) = sqrt (accuracy(2));
	coefficients{end + 1} = model.sv_coef;
	SVs{end + 1} = model.SVs;
	b{end + 1} = - model.rho;
end


%% Black box model, RBF (Radial Basis Function)
if ismember(3, MODELS_CHOSEN)
	fprintf('\nTraining model with RBF SVR\n');
	fflush(stdout);
	SVR_DESCRIPTIONS{end + 1} = 'Radial Basis Function SVR';

	[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 2 -q -h 0', C_range, E_range);
	options = ['-s 3 -t 2 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
	model = svmtrain (y_tr, X_tr, options);

	[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet
	sum_abs = 0;
	sum_rel = 0;
	for i = 1:N_test
		sum_abs += abs(y_test(i) - predictions(i, end));
		sum_rel += abs(y_test(i) - predictions(i, end) / predictions(i, end));
	end
	mean_abs = sum_abs / N_test;
	mean_rel = sum_rel / N_test;
	fprintf('\n Testing results:\n');
	fprintf('   RMSE = %f\n', sqrt(accuracy(2)));
	fprintf('   R^2 = %f\n', accuracy(3));
	fprintf('   Mean abs error = %f\n', mean_abs);
	fprintf('   Mean rel error = %f\n\n', mean_rel);

	models{end + 1} = model;
	Cs(end + 1) = C;
	Es(end + 1) = eps;
	RMSEs(end + 1) = sqrt (accuracy(2));
	coefficients{end + 1} = model.sv_coef;
	SVs{end + 1} = model.SVs;
	b{end + 1} = - model.rho;
end


%% Linear Regression

if LINEAR_REGRESSION
	fprintf('\nLinear regression:\n');

	X_tr = [ones(N_train, 1) , X_tr]; %% Add the intercept

	[theta, ~, ~, ~, results] = regress(y_tr, X_tr);

	%% Print training results
	% fprintf('\n theta: \n'); disp(theta');
	% fprintf('\n Training results: \n');
	% fprintf('   R^2 = %f\n', results(1));
	% fprintf('   F = %f\n', results(2));
	% fprintf('   p-value = %f\n', results(3));
	% fprintf('   MSE = %f\n', results(4));

	predictions(:, end+1) = [ones(N_test, 1) X_test] * theta;

	% fprintf('predictions  real values\n');
	% disp([predictions(1:10,1) y_test(1:10)]);

	y_mean = mean(y_test);
	sum_residual = 0;
	sum_total = 0;
	sum_abs = 0;
	sum_rel = 0;
	for i = 1:N_test
		sum_residual += (y_test(i) - predictions(i, end))^2;
		sum_total += (y_test(i) - y_mean)^2;
		sum_abs += abs(y_test(i) - predictions(i, end));
		sum_rel += abs(y_test(i) - predictions(i, end) / predictions(i, end));
	end

	lin_RMSE = sqrt(sum_residual / N_test);	%% Root Mean Squared Error
	lin_R2 = 1 - (sum_residual / sum_total);		%% R^2
	lin_mean_abs = sum_abs / N_test;
	lin_mean_rel = sum_rel / N_test;

	fprintf('\n Testing results:\n');
	fprintf('   RMSE = %f\n', lin_RMSE);
	fprintf('   R^2 = %f\n', lin_R2);
	fprintf('   Mean abs error = %f\n', lin_mean_abs);
	fprintf('   Mean rel error = %f\n\n', lin_mean_rel);

	RMSEs(end + 1) = lin_RMSE; 

	b{end+1} = 0;
	coefficients{end + 1} = 0;
	SVs{end + 1} = 0;
	Cs(end + 1) = 0;
	Es(end + 1) = 0;

	% Removes the intercept
	X_tr = X_tr(:, 2:end);
end


%% PLOTTING SVR vs LR

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

	% plot(x, w(col)*x + b{svr_index}, 'g');
	if LINEAR_REGRESSION
		plot(x, x*theta(col+1), 'm', 'linewidth', 1);
	end

	for index = 1:length(MODELS_CHOSEN)
		[ysvr, ~, ~] = svmpredict(zeros(length(x), 1), xsvr, models{index}, '-q');	%% quiet
		plot(x, ysvr, COLORS{index}, 'linewidth', 1);
	end
	
	labels = {'Training set', 'Testing set'};
	if LINEAR_REGRESSION
		labels{end+1} = 'Linear regression';
	end
	labels(end+1:end+length(SVR_DESCRIPTIONS)) = SVR_DESCRIPTIONS;
	legend(labels, 'location', 'northwest');

	% Display SVR margins
	% plot(x, w(col)*x + b{svr_index} + Es(svr_index), 'k');
	% plot(x, w(col)*x + b{svr_index} - Es(svr_index), 'k');

	% Labels the axes
	xlabel(FEATURES_DESCRIPTIONS(col));
	ylabel('Completion Time');
	% title(cstrcat('Linear regression vs ', SVR_DESCRIPTIONS{svr_index})); 
	
	hold off;
	% pause;

end