%% Skeleton

%%% TODO
%% - Automatic feature selection (with penalties?)


clear all;
clc;
close all hidden;

addpath('./utility/');

BASE_DIR = './dati/';

%% List of all directories with train data
TRAIN_DATA_LOCATION = {'Query R/R2/Core/60','Query R/R2/Core/72','Query R/R2/Core/90','Query R/R2/Core/100','Query R/R2/Core/120'};
% TRAIN_DATA_LOCATION = {'Core/60', 'Core/80', 'Core/100', 'Core/120', 'Core/72'};

%% List of all directories with test data (leave {} if test data equals train data)
TEST_DATA_LOCATION = {'Query R/R2/Core/80'};
%TEST_DATA_LOCATION = {};

SAVE_DATA = true;
OUTPUT_FOLDER = 'outputPlots/R2/72CORE_VS_ALL/';
OUTPUT_FORMATS = {	{'-deps', '.eps'},					% generates only one .eps file black and white
					{'-depslatex', '.eps'},				% generates one .eps file containing only the plot and a .tex file that includes the plot and fill the legend with plain text
					{'-depsc', '.eps'},					% generates only one .eps file with colour
					{'-dpdflatex', '.pdf'}				% generates one .pdf file containing only the plot and a .tex file that includes the plot and fill the legend with plain text
					{'-dpdf', '.pdf'}					% generates one complete .pdf file A4
				};
PLOT_SAVE_FORMAT = 3;



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

FEATURES = [1:8, 12:13]; 			% All the features except Users and Bandwidth
% NB: Bandwidth crea problemi con la linear regression, perché i valori sono tutti uguali per alcuni test
% LINEAR_REGRESSION va messa a false in quei casi

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
COLORS = {'m', [1, 0.5, 0], 'c'};	% magenta, orange, cyan

LINEAR_REGRESSION = true;

TEST_ON_CORES = true;	% To add the "difference between means" metric

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
	tmp = train_data(:, 2:end);
	train_data = [train_data(:, 1) , tmp(:, FEATURES)];
	FEATURES_DESCRIPTIONS = FEATURES_DESCRIPTIONS(FEATURES);
end

test_data = [];
if not (isempty(TEST_DATA_LOCATION))
	test_data = get_all_data_from_dirs(BASE_DIR, TEST_DATA_LOCATION);
	if CHOOSE_FEATURES
		tmp = test_data(:, 2:end);
		test_data = [test_data(:, 1) , tmp(:, FEATURES)];
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
means = [];
R_2 = [];

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

	SVR_DESCRIPTIONS{end + 1} = 'Linear SVR';

	[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 0 -q -h 0', C_range, E_range);
	options = ['-s 3 -t 0 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
	model = svmtrain (y_tr, X_tr, options);

	[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet


	models{end + 1} = model;
	Cs(end + 1) = C;
	Es(end + 1) = eps;
	RMSEs(end + 1) = sqrt (accuracy(2));
	coefficients{end + 1} = model.sv_coef;
	SVs{end + 1} = model.SVs;
	b{end + 1} = - model.rho;
	R_2(end + 1) = accuracy(3);
end


%% Black box model, Polynomial
if ismember(2, MODELS_CHOSEN)
	fprintf('\nTraining model with polynomial SVR\n');
	%fflush(stdout);
	SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR';

	[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 1 -q -h 0', C_range, E_range);
	options = ['-s 3 -t 1 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
	model = svmtrain (y_tr, X_tr, options);

	[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

	models{end + 1} = model;
	Cs(end + 1) = C;
	Es(end + 1) = eps;
	RMSEs(end + 1) = sqrt (accuracy(2));
	coefficients{end + 1} = model.sv_coef;
	SVs{end + 1} = model.SVs;
	b{end + 1} = - model.rho;
	R_2(end + 1) = accuracy(3);
end


%% Black box model, RBF (Radial Basis Function)
if ismember(3, MODELS_CHOSEN)
	fprintf('\nTraining model with RBF SVR\n');
	%fflush(stdout);
	SVR_DESCRIPTIONS{end + 1} = 'Radial Basis Function SVR';

	[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 2 -q -h 0', C_range, E_range);
	options = ['-s 3 -t 2 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
	model = svmtrain (y_tr, X_tr, options);

	[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet


	models{end + 1} = model;
	Cs(end + 1) = C;
	Es(end + 1) = eps;
	RMSEs(end + 1) = sqrt (accuracy(2));
	coefficients{end + 1} = model.sv_coef;
	SVs{end + 1} = model.SVs;
	b{end + 1} = - model.rho;
	R_2(end + 1) = accuracy(3);
end


%% Linear Regression

if LINEAR_REGRESSION
	fprintf('\nLinear regression:\n');

	X_tr = [ones(N_train, 1) , X_tr]; %% Add the intercept

	[theta, ~, ~, ~, results] = regress(y_tr, X_tr);

	predictions(:, end+1) = [ones(N_test, 1) X_test] * theta;

	models{end + 1} = {};
	Cs(end + 1) = 0;
	Es(end + 1) = 0;
	RMSEs(end + 1) = -1;			%% Will be computed later 
	coefficients{end + 1} = 0;
	SVs{end + 1} = 0;
	b{end+1} = 0;
	R_2(end + 1) = -1;				%% Will be computed later

	% Removes the intercept
	X_tr = X_tr(:, 2:end);
end


%% Creates the directory for saving plots and data
%% @TODO this doesn't work for multiple directories
fd = -1;
if SAVE_DATA
	if ~ exist(OUTPUT_FOLDER)
		if ~ mkdir(OUTPUT_FOLDER)
			fprintf('[ERROR] Could not create output folder\nCreate the output folder first and then restart this script\n');
			quit;
		end
	end

	results_filename = strcat(OUTPUT_FOLDER, 'report.txt');
	fd = fopen(results_filename, 'w');

end


%% Compute metrics for all models

for index = 1:length(MODELS_CHOSEN)
	sum_abs = sum(abs(y_test .- predictions(:, index)));
	sum_rel = sum(abs((y_test .- predictions(:, index)) ./ predictions(:, index)));
	

	mean_abs = sum_abs / N_test;
	mean_rel = sum_rel / N_test;
	fprintf('\n Testing results for %s:\n', SVR_DESCRIPTIONS{index});
	fprintf('   RMSE = %f\n', RMSEs(index));
	fprintf('   R^2 = %f\n', R_2(index));
	fprintf('   Mean abs error = %f\n', mean_abs);
	fprintf('   Mean rel error = %f\n', mean_rel);

	if SAVE_DATA
		fprintf(fd, '\n Testing results for %s:\n', SVR_DESCRIPTIONS{index});
		fprintf(fd, '   RMSE = %f\n', RMSEs(index));
		fprintf(fd, '   R^2 = %f\n', R_2(index));
		fprintf(fd, '   Mean abs error = %f\n', mean_abs);
		fprintf(fd, '   Mean rel error = %f\n', mean_rel);
	end


	y_mean = mean(y_test);
	pred_mean = mean(predictions(:, index));
	means(end + 1) = pred_mean;
	if TEST_ON_CORES
		diff_means = pred_mean - y_mean;
		fprintf('   Difference between means = %f\n', diff_means);
		if SAVE_DATA
			fprintf(fd, '   Difference between means = %f\n', diff_means);
		end
	end
	fprintf('\n');
end

if LINEAR_REGRESSION
	y_mean = mean(y_test);

	sum_residual = sum((y_test .- predictions(:, end)).^2);
	sum_total = sum((y_test .- y_mean).^2);
	sum_abs = sum(abs(y_test - predictions(:, end)));
	sum_rel = sum(abs((y_test - predictions(:, end)) ./ predictions(:, end)));

	lin_RMSE = sqrt(sum_residual / N_test);			% Root Mean Squared Error
	lin_R2 = 1 - (sum_residual / sum_total);		% R^2
	lin_mean_abs = sum_abs / N_test;
	lin_mean_rel = sum_rel / N_test;

	fprintf('\n Testing results for linear regression:\n');
	fprintf('   RMSE = %f\n', lin_RMSE);
	fprintf('   R^2 = %f\n', lin_R2);
	fprintf('   Mean abs error = %f\n', lin_mean_abs);
	fprintf('   Mean rel error = %f\n', lin_mean_rel);

	if SAVE_DATA
		fprintf(fd, '\n Testing results for linear regression:\n');
		fprintf(fd, '   RMSE = %f\n', lin_RMSE);
		fprintf(fd, '   R^2 = %f\n', lin_R2);
		fprintf(fd, '   Mean abs error = %f\n', lin_mean_abs);
		fprintf(fd, '   Mean rel error = %f\n', lin_mean_rel);
	end

	RMSEs(end) = lin_RMSE;
	R_2(end) = lin_R2;

	pred_mean = mean(predictions(:, end));
	means(end + 1) = pred_mean;
	if TEST_ON_CORES
		diff_means = pred_mean - y_mean;
		fprintf('   Difference between means = %f\n', diff_means);
		if SAVE_DATA
			fprintf(fd, '   Difference between means = %f\n', diff_means);
		end
	end
	fprintf('\n');
end


%% Closes the file descriptor
if SAVE_DATA
	fclose(fd);
end





% Denormalize means
means = (means * sigma_y) + mu_y;




%% Denormalize features

if NORMALIZE_FEATURE
	X_tr_denorm = X_tr .* (ones(N_train, 1) * sigma_X) .+ (ones(N_train, 1) * mu_X);
	y_tr_denorm = y_tr * sigma_y + mu_y;
	X_test_denorm = X_test .* (ones(N_test, 1) * sigma_X) .+ (ones(N_test, 1) * mu_X);
	y_test_denorm = y_test * sigma_y + mu_y;
else
	X_tr_denorm = X_tr;
	y_tr_denorm = y_tr;
	X_test_denorm = X_test;
	y_test_denorm = y_test;
end




%% PLOTTING SVR vs LR

for col = 1:M

	figure;
	hold on;

	scatter(X_tr_denorm(:, col), y_tr_denorm, 'r', 'x');
	scatter(X_test_denorm(:, col), y_test_denorm, 'b');

	% x = linspace(min(X_test(:, col)), max(X_test(:, col)));		% Normalized, we need this for the predictions
	x = linspace(min(min(X_test(:, col)), min(X_tr(:, col))), max(max(X_test(:, col)), max(X_tr(:, col))));  %% fill all the plot
	x_denorm = (x * sigma_X(col)) + mu_X(col);

	xsvr = zeros(length(x), M);			% xsvr is a matrix of zeros, except for the column we're plotting currently
	xsvr(:, col) = x;					% It must be normalized to use svmpredict with the SVR models we found


	if LINEAR_REGRESSION
		ylin = x * theta(col+1);

		% Denormalize y
		if NORMALIZE_FEATURE
			ylin = (ylin * sigma_y) + mu_y;
		end

		% if (ismember(13, FEATURES) & (col == M))   %% If we are plotting N-cores
		% 	scatter(X_test_denorm(1, col), means(end), 10, 'g', 'd', 'filled');
		% end
		plot(x_denorm, ylin, 'g', 'linewidth', 1);
		

	end

	for index = 1:length(MODELS_CHOSEN)
		[ysvr, ~, ~] = svmpredict(zeros(length(x), 1), xsvr, models{index}, '-q');	%% quiet

		% Denormalize
		if NORMALIZE_FEATURE
			ysvr = (ysvr * sigma_y) + mu_y;
		end 

		% if (ismember(13, FEATURES) & (col == M))
		% 	scatter(X_test_denorm(1, col), means(index), 10, COLORS{index}, 'd', 'filled');
		% end	
		plot(x_denorm, ysvr, 'color', COLORS{index}, 'linewidth', 1);
		

	end

	% Plot the mean of the test values (for nCores)
	if (ismember(13, FEATURES) & (col == M))
		scatter(X_test_denorm(1, col), mean(y_test_denorm), 10, 'k', '.');		
	end
	
	labels = {'Training set', 'Testing set'};
	if LINEAR_REGRESSION
		labels{end+1} = 'Linear regression';
	end
	labels(end+1:end+length(SVR_DESCRIPTIONS)) = SVR_DESCRIPTIONS;
	legend(labels, 'location', 'northeastoutside');

	

	% Labels the axes
	xlabel(FEATURES_DESCRIPTIONS{col});
	ylabel('Completion Time');
	% title(cstrcat('Linear regression vs ', SVR_DESCRIPTIONS{svr_index})); 
	if SAVE_DATA
		% NOTE: the file location shouldn't have any spaces
		file_location = strrep(strcat(OUTPUT_FOLDER, 'plot_', FEATURES_DESCRIPTIONS{col}, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
		print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
	end
	hold off;
	
	% pause;

end