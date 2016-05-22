%% Skeleton

%%% TODO
%% - Automatic feature selection (with penalties?)

QUERIES = {'R1', 'R2', 'R3', 'R4', 'R5'};
DATASIZES = {'250', '500', '750', '1000'};

for query_id = 1:length(QUERIES)

	for datasize_id = 1:length(DATASIZES)

		clear all;
		clc;
		close all hidden;

		addpath('./utility/');

		BASE_DIR = './dati/Query R/';

		QUERY = QUERIES{query_id};
		DATASIZE = DATASIZES{datasize_id};

		%% List of all directories with train data
		TRAIN_DATA_LOCATION = {strcat(QUERY, '/Datasize/', DATASIZE)};
		% TRAIN_DATA_LOCATION = {strcat('')};
		% TRAIN_DATA_LOCATION = {'Core/60', 'Core/80', 'Core/100', 'Core/120', 'Core/72'};
		% TRAIN_DATA_LOCATION = {'Query R/R1/Datasize/750'};

		%% List of all directories with test data (leave {} if test data equals train data)
		% TEST_DATA_LOCATION = {'Query R/R1/Core/120'};
		TEST_DATA_LOCATION = {};

		OUTPUT_LATEX = true;
		TABLE_CAPTION = cstrcat('Results for ', QUERY, '-', DATASIZE);
		PLOT_CAPTION = cstrcat('Completion time vs ncores for query ', QUERY, ' with datasize ', DATASIZE);
		TABLE_LABEL = cstrcat('tab:', 'coreonly_linear_', QUERY, '_', DATASIZE);
		PLOT_LABEL = cstrcat('fig:', 'coreonly_linear_', QUERY, '_', DATASIZE);

		SAVE_DATA = false;

		% OUTPUT_FOLDER = strcat('output/', QUERY, '_ALL_FEATURES/');
		% OUTPUT_FOLDER = strcat('output/', QUERY, '_', DATASIZE, '_ONLY_1_LINEAR_NCORE/');
		OUTPUT_FOLDER = strcat('output/TESTING/');

		OUTPUT_FORMATS = {	{'-deps', '.eps'},					% generates only one .eps file black and white
							{'-depslatex', '.eps'},				% generates one .eps file containing only the plot and a .tex file that includes the plot and fill the legend with plain text
							{'-depsc', '.eps'},					% generates only one .eps file with colour
							{'-dpdflatex', '.pdf'}				% generates one .pdf file containing only the plot and a .tex file that includes the plot and fill the legend with plain text
							{'-dpdf', '.pdf'}					% generates one complete .pdf file A4
						};
		PLOT_SAVE_FORMAT = 3;

		ENABLE_FEATURE_FILTERING = false;
		COMPLETION_TIME_THRESHOLD = 32000;


		%% CHANGE THESE IF TEST == TRAIN
		TRAIN_FRAC_WO_TEST = 0.6;
		TEST_FRAC_WO_TEST = 0.2;

		%% CHANGE THESE IF TEST != TRAIN
		TRAIN_FRAC_W_TEST = 0.7;


		N_CORES_INVERSE = false;		%% ncores^(-1)
		NORMALIZE_FEATURE = true;
		CLEAR_OUTLIERS = true;


		LEARNING_CURVES = false;

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
		% 12 -> Datasize
		% 13 -> N Core
		CHOOSE_FEATURES = true;

		FEATURES = [3:8,13];
		% FEATURES = [13];


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
			'Datasize',
			'N core'
		};

		%% Choose which SVR models to use
		% 1 -> Linear SVR
		% 2 -> Polynomial SVR (2 degree)
		% 3 -> Polynomial SVR (3 degree)
		% 4 -> Polynomial SVR (4 degree)
		% 5 -> Polynomial SVR (6 degree)
		% 6 -> RBF SVR
		MODELS_CHOSEN = [1, 2, 3, 4, 5, 6];
		COLORS = {'g', [1, 0.5, 0.2], 'c', 'k', 'm', 'r'};	% magenta, orange, cyan, black, green, red

		LINEAR_REGRESSION = true;

		BEST_MODELS = true;

		TEST_ON_CORES = false;	% To add the 'difference between means' metric

		rand('seed', 18);
		SHUFFLE_DATA = true;

		C_range = linspace (0.1, 5, 20);
		E_range = linspace (0.1, 5, 20);



		% --------------------------------------------------------------------------------------------------
		% |									       DO NOT  MODIFY 								           |
		% |										   UNDER THIS BOX 								           |
		% --------------------------------------------------------------------------------------------------

		% Create output folder
		if ~ exist(OUTPUT_FOLDER)		%% Checks if the folder exists
			if ~ mkdir(OUTPUT_FOLDER)		%% Try with the mkdir function
				if system(cstrcat('mkdir -p ', OUTPUT_FOLDER))		%% This creates subfolders
					fprintf('[ERROR] Could not create output folder\nCreate the output folder first and then restart this script\n');
					quit;
				end
			end
		end



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


		if ENABLE_FEATURE_FILTERING
			rows_ok = train_data(:, 1) < COMPLETION_TIME_THRESHOLD;
			train_data = train_data(rows_ok, :);

			if not (isempty(TEST_DATA_LOCATION))
				rows_ok = test_data(:, 1) < COMPLETION_TIME_THRESHOLD;
				test_data = test_data(rows_ok, :);
			end
		end



		M = size(train_data, 2) - 1;   		%% Number of features
		N_train = size(train_data, 1);		%% Number of train tuples
		N_test = size(test_data, 1);		%% Number of test tuples

		complete_data = [train_data ; test_data];


		if CLEAR_OUTLIERS
			% [clean, indices] = clear_outliers(complete_data);

			[clean,indices] = clear_outliers_ncores(complete_data);

			train_data = clean(indices <= N_train, :);
			test_data = clean(indices > N_train, :);

			N_train = size(train_data, 1);		%% Number of train tuples
			N_test = size(test_data, 1);		%% Number of test tuples

			complete_data = [train_data ; test_data];
		end


		if N_CORES_INVERSE

			complete_data(:, end) = 1./complete_data(:, end);  %% replace nCores with 1/nCores

		end


		mu = zeros(M+1, 1);
		sigma = ones(M+1, 1);

		if NORMALIZE_FEATURE
			[scaled, mu, sigma] = zscore(complete_data);

			train_data = scaled(1:N_train, :);
			test_data = scaled(N_train+1:end, :);

			% Save data for - maybe - later uses
			save(strcat(OUTPUT_FOLDER, 'mu_sigma.mat'), 'mu', 'sigma');

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

		Cs = [];     
		Es = [];
		predictions = [];
		coefficients = {};
		SVs = {};
		b = {};
		SVR_DESCRIPTIONS = {};
		models = {};
		means = [];

		% Saving test metrics
		RMSEs = [];
		R_2 = [];
		MAE = [];	% Mean absolute error
		MRE = [];	% Mean relative error
		DM = [];	% Difference between means

		%% SVR

		% svmtrain parameters
		% -s --> SVM type (3 = epsilon-SVR)
		% -t --> kernel tyle (0 = linear, 1 = polynomial, 2 = gaussian, 3 = sigmoid)
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

			if LEARNING_CURVES
				[m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
				h = plot_learning_curves(m, mse_train, mse_test);
				print('-depsc', cstrcat(OUTPUT_FOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
				close(h);
			end

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
			fprintf('\nTraining model with polynomial(2) SVR\n');
			%fflush(stdout);
			SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (2)';

			[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 2 -t 1 -q -h 0', C_range, E_range);
			options = ['-s 3 -d 2 -t 1 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
			model = svmtrain (y_tr, X_tr, options);

			[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

			if LEARNING_CURVES
				[m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
				h = plot_learning_curves(m, mse_train, mse_test);
				print('-depsc', cstrcat(OUTPUT_FOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
				close(h);
			end

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
		if ismember(3, MODELS_CHOSEN)
			fprintf('\nTraining model with polynomial(3) SVR\n');
			%fflush(stdout);
			SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (3)';

			[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 3 -t 1 -q -h 0', C_range, E_range);
			options = ['-s 3 -d 3 -t 1 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
			model = svmtrain (y_tr, X_tr, options);

			[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

			if LEARNING_CURVES
				[m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
				h = plot_learning_curves(m, mse_train, mse_test);
				print('-depsc', cstrcat(OUTPUT_FOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
				close(h);
			end

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
		if ismember(4, MODELS_CHOSEN)
			fprintf('\nTraining model with polynomial(4) SVR\n');
			%fflush(stdout);
			SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (4)';

			[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 4 -t 1 -q -h 0', C_range, E_range);
			options = ['-s 3 -d 4 -t 1 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
			model = svmtrain (y_tr, X_tr, options);

			[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

			if LEARNING_CURVES
				[m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
				h = plot_learning_curves(m, mse_train, mse_test);
				print('-depsc', cstrcat(OUTPUT_FOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
				close(h);
			end

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
		if ismember(5, MODELS_CHOSEN)
			fprintf('\nTraining model with polynomial(6) SVR\n');
			%fflush(stdout);
			SVR_DESCRIPTIONS{end + 1} = 'Polynomial SVR (6)';

			[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -d 6 -t 1 -q -h 0', C_range, E_range);
			options = ['-s 3 -d 6 -t 1 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
			model = svmtrain (y_tr, X_tr, options);

			[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

			if LEARNING_CURVES
				[m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
				h = plot_learning_curves(m, mse_train, mse_test);
				print('-depsc', cstrcat(OUTPUT_FOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
				close(h);
			end

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
		if ismember(6, MODELS_CHOSEN)
			fprintf('\nTraining model with RBF SVR\n');
			%fflush(stdout);
			SVR_DESCRIPTIONS{end + 1} = 'Gaussian SVR';

			[C, eps] = model_selection (y_tr, X_tr, y_cv, X_cv, '-s 3 -t 2 -q -h 0', C_range, E_range);
			options = ['-s 3 -t 2 -h 0 -p ', num2str(eps), ' -c ', num2str(C)];
			model = svmtrain (y_tr, X_tr, options);

			[predictions(:, end + 1), accuracy, ~] = svmpredict (y_test, X_test, model, '-q');  %% quiet

			if LEARNING_CURVES
				[m, mse_train, mse_test] = learning_curves(y_tr, X_tr, y_test, X_test, [options, ' -q']);
				h = plot_learning_curves(m, mse_train, mse_test);
				print('-depsc', cstrcat(OUTPUT_FOLDER, 'learning_curve_', SVR_DESCRIPTIONS{end}, '.eps'));
				close(h);
			end

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
			fprintf('\nTraining Linear regression.\n');

			X_tr = [ones(N_train, 1) , X_tr]; %% Add the intercept

			[theta, ~, ~, ~, results] = regress(y_tr, X_tr);

			predictions(:, end+1) = [ones(N_test, 1) X_test] * theta;

			models{end + 1} = {};
			Cs(end + 1) = 0;
			Es(end + 1) = 0;
			% RMSEs(end + 1) = -1;			%% Will be computed later 
			coefficients{end + 1} = 0;
			SVs{end + 1} = 0;
			b{end+1} = 0;
			% R_2(end + 1) = -1;				%% Will be computed later

			% Removes the intercept
			X_tr = X_tr(:, 2:end);
		end



		fd = -1;
		if SAVE_DATA

			results_filename = strcat(OUTPUT_FOLDER, 'report.txt');
			fd = fopen(results_filename, 'w');

			%% Prints train and test data location

			fprintf(fd, 'TRAIN DATA:\n');
			for index = 1:length(TRAIN_DATA_LOCATION)
				fprintf(fd, '%s\n', TRAIN_DATA_LOCATION{index});
			end

			fprintf(fd, '\n\nTEST DATA:\n');
			for index = 1:length(TEST_DATA_LOCATION)
				fprintf(fd, '%s\n', TEST_DATA_LOCATION{index});
			end

			fprintf(fd, '\n\n\n');
		end

		if OUTPUT_LATEX
			if ~ exist(OUTPUT_FOLDER)		%% Checks if the folder exists
				if ~ mkdir(OUTPUT_FOLDER)		%% Try with the mkdir function
					if system(cstrcat('mkdir -p ', OUTPUT_FOLDER))		%% This creates subfolders
						fprintf('[ERROR] Could not create output folder\nCreate the output folder first and then restart this script\n');
						quit;
					end
				end
			end
			latex_filename_table = strcat(OUTPUT_FOLDER, 'outputlatex_table.txt');
			flatex_table = fopen(latex_filename_table, 'w');

			latex_filename_plot = strcat(OUTPUT_FOLDER, 'outputlatex_plot.txt');
			flatex_plot = fopen(latex_filename_plot, 'w');

			if BEST_MODELS
				latex_filename_plot_bestmodels = strcat(OUTPUT_FOLDER, 'outputlatex_plot_bestmodels.txt');
				flatex_plot_bestmodels = fopen(latex_filename_plot_bestmodels, 'w');
			end

			fprintf(flatex_table, cstrcat('\\begin{table}[H]\n', ...
							'\\centering\n', ...
							'\\begin{adjustbox}{center}\n', ...
							'\\begin{tabular}{c | c M{1.2cm} M{2.5cm} M{2.5cm} M{1.8cm}}\n'));
			if TEST_ON_CORES
				fprintf(flatex_table, 'Model & RMSE & R\\textsuperscript{2} & Mean absolute error & Mean relative error & Mean difference \\tabularnewline\n');
			else
				fprintf(flatex_table, 'Model & RMSE & R\\textsuperscript{2} & Mean absolute error & Mean relative error \\tabularnewline\n');
			end

			fprintf(flatex_table, '\\hline\n');

		end	

		%% Compute metrics for all models

		if LINEAR_REGRESSION
			y_mean = mean(y_test);

			sum_residual = sum((y_test - predictions(:, end)).^2);
			sum_total = sum((y_test - y_mean).^2);

			real_test_values = mu_y + sigma_y * y_test;
			real_predictions = mu_y + sigma_y * predictions(:, end);

			abs_err = abs(real_test_values - real_predictions);
			rel_err = abs_err ./ real_test_values;

			lin_mean_abs = mean(abs_err);
			lin_mean_rel = mean(rel_err);


			% sum_abs = sum(abs(y_test - predictions(:, end)));
			% sum_rel = sum(sigma_y * abs((y_test - predictions(:, end)) ./ (sigma_y * predictions(:, end)) + mu_y);

			lin_RMSE = sqrt(sum_residual / N_test);			% Root Mean Squared Error
			lin_R2 = 1 - (sum_residual / sum_total);		% R^2
			% lin_mean_abs = ((sum_abs / N_test));
			% lin_mean_rel = sum_rel / N_test;

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

			RMSEs(end + 1) = lin_RMSE;
			R_2(end + 1) = lin_R2;

			pred_mean = mean(predictions(:, end));
			means(end + 1) = pred_mean;
			if TEST_ON_CORES
				diff_means = pred_mean - y_mean;
				fprintf('   Difference between means = %f\n', diff_means);
				if SAVE_DATA
					fprintf(fd, '   Difference between means = %f\n', diff_means);
				end
			end


			if (OUTPUT_LATEX & ~TEST_ON_CORES)
				fprintf(flatex_table, 'Linear regression & %5.4f & %5.4f & %6.0f & %5.4f \\\\\n', lin_RMSE, lin_R2, lin_mean_abs, lin_mean_rel);
			end

			if (OUTPUT_LATEX & TEST_ON_CORES)
				fprintf(flatex_table, 'Linear regression & %5.4f & %5.4f & %6.0f & %5.4f & %5.4f \\\\\n', lin_RMSE, lin_R2, lin_mean_abs, lin_mean_rel, diff_means);
			end

			fprintf('\n');
		end


		for index = 1:length(MODELS_CHOSEN)
			real_predictions = mu_y + sigma_y * predictions(:, index);
			real_test_values = mu_y + sigma_y * y_test;

			abs_err = abs(real_predictions - real_test_values);
			rel_err = abs_err ./ real_test_values;

			mean_abs = mean(abs_err);
			mean_rel = mean(rel_err);

			% sum_abs = sum(abs(y_test - predictions(:, index)));
			% sum_rel = sum(sigma_y * abs((y_test - predictions(:, index)) ./ (sigma_y * predictions(:, index)) + mu_y);

			% mean_abs = ((sum_abs / N_test) * sigma_y);
			% mean_rel = sum_rel / N_test;
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

			if (OUTPUT_LATEX & ~TEST_ON_CORES)
				fprintf(flatex_table, '%s & %5.4f & %5.4f & %6.0f & %5.4f \\\\\n', SVR_DESCRIPTIONS{index}, RMSEs(index), R_2(index), mean_abs, mean_rel);
			end

			if (OUTPUT_LATEX & TEST_ON_CORES)
				fprintf(flatex_table, '%s & %5.4f & %5.4f & %6.0f & %5.4f & %5.4f \\\\\n', SVR_DESCRIPTIONS{index}, RMSEs(index), R_2(index), mean_abs, mean_rel, diff_means);
			end

			fprintf('\n');
		end

		if OUTPUT_LATEX
			fprintf(flatex_table, cstrcat('\\end{tabular}\n', ...
										'\\end{adjustbox}\n', ...
										'\\\\\n', ...
										'\\caption{', TABLE_CAPTION, '}\n', ...
										'\\label{', TABLE_LABEL, '}\n', ...
										'\\end{table}\n'));
			fclose(flatex_table);

			fprintf(flatex_plot, cstrcat('\n\\begin {figure}[hbtp]\n', ...
										'\\centering\n', ...
										'\\includegraphics[width=\\textwidth]{', OUTPUT_FOLDER, 'plot_', QUERY, '_', DATASIZE, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}, '}\n', ...
										'\\caption{', PLOT_CAPTION, '}\n', ...
										'\\label{', PLOT_LABEL, '}\n', ...
										'\\end {figure}\n'));	
			fclose(flatex_plot);

			if BEST_MODELS
				fprintf(flatex_plot_bestmodels, cstrcat('\n\\begin {figure}[hbtp]\n', ...
														'\\centering\n', ...
														'\\includegraphics[width=\\textwidth]{', OUTPUT_FOLDER, 'plot_', QUERY, '_', DATASIZE, '_bestmodels', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}, '}\n', ...
														'\\caption{', PLOT_CAPTION, '}\n', ...
														'\\label{', PLOT_LABEL, '}\n', ...
														'\\end {figure}\n'));	
				fclose(flatex_plot_bestmodels);
			end
		end

		%% Stores the context and closes the file descriptor
		if SAVE_DATA
			fprintf(fd, '\n\n\n========================\n\n\n');
			fprintf(fd, 'ENABLE_FEATURE_FILTERING: %d\n', ENABLE_FEATURE_FILTERING);
			fprintf(fd, 'COMPLETION_TIME_THRESHOLD: %d\n', COMPLETION_TIME_THRESHOLD);
			fprintf(fd, 'TRAIN_FRAC_WO_TEST: %f\n', TRAIN_FRAC_WO_TEST);
			fprintf(fd, 'TEST_FRAC_WO_TEST: %f\n', TEST_FRAC_WO_TEST);
			fprintf(fd, 'TRAIN_FRAC_W_TEST: %f\n', TRAIN_FRAC_W_TEST);
			fprintf(fd, 'NORMALIZE_FEATURE: %d\n', NORMALIZE_FEATURE);
			fprintf(fd, 'CLEAR_OUTLIERS: %d\n', CLEAR_OUTLIERS);
			fprintf(fd, 'CHOOSE_FEATURES: %d\n', CHOOSE_FEATURES);
			fprintf(fd, 'FEATURES: %s --> ', mat2str(FEATURES));
			for id = 1:length(FEATURES)
				fprintf(fd, '%s   ', FEATURES_DESCRIPTIONS{id});
			end
			fprintf(fd, '\n');
			fprintf(fd, 'TEST_ON_CORES: %d\n', TEST_ON_CORES);
			fprintf(fd, 'SHUFFLE_DATA: %d\n', SHUFFLE_DATA);
			save(strcat(OUTPUT_FOLDER, 'models.mat'), 'SVs', 'coefficients', 'b', 'models', 'Cs', 'Es', 'theta', 'mu', 'sigma');
			

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

		%% Determine the best 3 models
		if BEST_MODELS
			tempR_2 = R_2;
			best_models_idx = [];
			[~, best_models_idx(end+1)] = max(tempR_2);
			tempR_2(best_models_idx(end)) = -1;
			[~, best_models_idx(end+1)] = max(tempR_2);
			tempR_2(best_models_idx(end)) = -1;	
			[~, best_models_idx(end+1)] = max(tempR_2);
		end


		%% PLOTTING SVR vs LR

		for col = 1:M

			figure;
			hold on;

			% scatter(X_tr_denorm(:, col), y_tr_denorm, 'r', 'x');
			% scatter(X_test_denorm(:, col), y_test_denorm, 'b');
			X_tr_denorm_col = X_tr_denorm(:, col);
			X_test_denorm_col = X_test_denorm(:, col);
			
			if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
				X_tr_denorm_col = 1./X_tr_denorm_col;
				X_test_denorm_col = 1./X_test_denorm_col;
			end

			my_scatter(X_tr_denorm_col, y_tr_denorm, 'r', 'x');
			my_scatter(X_test_denorm_col, y_test_denorm, 'b');


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

				x_plot = x_denorm;
				if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
					x_plot = 1./x_plot;  
				end

				plot(x_plot, ylin, 'color', [0.5, 0, 1], 'linewidth', 1);

				x = x_denorm;
				y = ylin;
				save(cstrcat(OUTPUT_FOLDER, 'Linear Regression.mat'), 'x', 'y', 'QUERY', 'DATASIZE');
				

			end

			for index = 1:length(MODELS_CHOSEN)
				[ysvr, ~, ~] = svmpredict(zeros(length(x), 1), xsvr, models{index}, '-q');	%% quiet

				% Denormalize
				if NORMALIZE_FEATURE
					ysvr = (ysvr * sigma_y) + mu_y;
				end 

				x_plot = x_denorm;
				if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
					x_plot = 1./x_plot;  
				end

				plot(x_plot, ysvr, 'color', COLORS{index}, 'linewidth', 1);

				x = x_denorm;
				y = ysvr;
				save(cstrcat(OUTPUT_FOLDER, SVR_DESCRIPTIONS{index}, '.mat'), 'x', 'y', 'QUERY', 'DATASIZE');

			end

			% Plot the mean of the test values (for nCores)
			% if (TEST_ON_CORES & ismember(13, FEATURES) & (col == M))
			% 	scatter(X_test_denorm(1, col), mean(y_test_denorm), 10, 'k', '.');		
			% end
			
			labels = {'Training set', 'Testing set'};
			if LINEAR_REGRESSION
				labels{end+1} = 'Linear Regression';
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
				% file_location = strrep(strcat(OUTPUT_FOLDER, 'plot_', QUERY, '_', DATASIZE, FEATURES_DESCRIPTIONS{col}, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
				print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
			end

			if SAVE_DATA & ismember(13, FEATURES) & (col == M)
				file_location = strrep(strcat(OUTPUT_FOLDER, 'plot_', QUERY, '_', DATASIZE, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
				print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
			end

			hold off;
			
			% pause;

		end


		%% Plot and save (only the best models)

		if BEST_MODELS
			for col = 1:M

				figure;
				hold on;

				% scatter(X_tr_denorm(:, col), y_tr_denorm, 'r', 'x');
				% scatter(X_test_denorm(:, col), y_test_denorm, 'b');
				X_tr_denorm_col = X_tr_denorm(:, col);
				X_test_denorm_col = X_test_denorm(:, col);
				
				if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
					X_tr_denorm_col = 1./X_tr_denorm_col;
					X_test_denorm_col = 1./X_test_denorm_col;
				end

				my_scatter(X_tr_denorm_col, y_tr_denorm, 'r', 'x');
				my_scatter(X_test_denorm_col, y_test_denorm, 'b');


				% x = linspace(min(X_test(:, col)), max(X_test(:, col)));		% Normalized, we need this for the predictions
				x = linspace(min(min(X_test(:, col)), min(X_tr(:, col))), max(max(X_test(:, col)), max(X_tr(:, col))));  %% fill all the plot
				x_denorm = (x * sigma_X(col)) + mu_X(col);

				xsvr = zeros(length(x), M);			% xsvr is a matrix of zeros, except for the column we're plotting currently
				xsvr(:, col) = x;					% It must be normalized to use svmpredict with the SVR models we found


				if (LINEAR_REGRESSION && ismember(length(MODELS_CHOSEN)+1, best_models_idx))	% if linear regression is one of the best models
					ylin = x * theta(col+1);

					% Denormalize y
					if NORMALIZE_FEATURE
						ylin = (ylin * sigma_y) + mu_y;
					end

					x_plot = x_denorm;
					if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
						x_plot = 1./x_plot;  
					end

					plot(x_plot, ylin, 'color', [0.5, 0, 1], 'linewidth', 1);		

				end

				for index = 1:length(MODELS_CHOSEN)
					if ismember(index, best_models_idx)
						[ysvr, ~, ~] = svmpredict(zeros(length(x), 1), xsvr, models{index}, '-q');	%% quiet

						% Denormalize
						if NORMALIZE_FEATURE
							ysvr = (ysvr * sigma_y) + mu_y;
						end 

						x_plot = x_denorm;
						if (N_CORES_INVERSE & ismember(13, FEATURES) & (col == M))
							x_plot = 1./x_plot;  
						end

						plot(x_plot, ysvr, 'color', COLORS{index}, 'linewidth', 1);
					end
				end

				% Plot the mean of the test values (for nCores)
				% if (TEST_ON_CORES & ismember(13, FEATURES) & (col == M))
				% 	scatter(X_test_denorm(1, col), mean(y_test_denorm), 10, 'k', '.');		
				% end
				
				labels = {'Training set', 'Testing set'};
				if (LINEAR_REGRESSION && ismember(length(MODELS_CHOSEN)+1, best_models_idx))
					labels{end+1} = 'Linear Regression';
				end
				for index = 1:length(SVR_DESCRIPTIONS)
					if ismember(index, best_models_idx)
						labels(end+1) = SVR_DESCRIPTIONS{index};
					end
				end
				legend(labels, 'location', 'northeastoutside');

				

				% Labels the axes
				xlabel(FEATURES_DESCRIPTIONS{col});
				ylabel('Completion Time');
				% title(cstrcat('Linear regression vs ', SVR_DESCRIPTIONS{svr_index})); 
				if SAVE_DATA
					% NOTE: the file location shouldn't have any spaces
					file_location = strrep(strcat(OUTPUT_FOLDER, 'plot_', FEATURES_DESCRIPTIONS{col}, '_bestmodels', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
					% file_location = strrep(strcat(OUTPUT_FOLDER, 'plot_', QUERY, '_', DATASIZE, FEATURES_DESCRIPTIONS{col}, OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
					print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
				end

				if SAVE_DATA & ismember(13, FEATURES) & (col == M)
					file_location = strrep(strcat(OUTPUT_FOLDER, 'plot_', QUERY, '_', DATASIZE, '_bestmodels', OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{2}), ' ', '');
					print(OUTPUT_FORMATS{PLOT_SAVE_FORMAT}{1}, file_location);
				end

				hold off;
				
				% pause;

			end
		end

	end

end