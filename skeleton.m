%% Skeleton

clear all;
clc;
close all hidden;


BASE_DIR = "/Users/Andrea/Documents/ProgettoML/dati/";

%% List of all directories with train data
% TRAIN_DATA_LOCATION = {"Query R/R1","Query R/R2","Query R/R5","Query R/R4"};
TRAIN_DATA_LOCATION = {"Core/60", "Core/80", "Core/100", "Core/90", "Core/120"};

%% List of all directories with test data (leave {} if test data equals train data)
% TEST_DATA_LOCATION = {"Query R/R3"};
TEST_DATA_LOCATION = {"Core/72"};

%% CHANGE THESE IF TEST == TRAIN
TRAIN_FRAC_WO_TEST = 0.6;
TEST_FRAC_WO_TEST = 0.2;

%% CHANGE THESE IF TEST != TRAIN
TRAIN_FRAC_W_TEST = 0.7;


NON_LINEAR_FEATURES = false;
NORMALIZE_FEATURE = true;
CLEAR_OUTLIERS = true;

SHUFFLE_DATA = true;

REMOVE_USERS = true;


%% Retrieve the data

train_data = get_all_data_from_dirs(BASE_DIR, TRAIN_DATA_LOCATION);
if REMOVE_USERS
	train_data = [train_data(:, 1:11), train_data(:, 13:end)];
end

test_data = [];
if not (isempty(TEST_DATA_LOCATION))
	test_data = get_all_data_from_dirs(BASE_DIR, TEST_DATA_LOCATION);
	if REMOVE_USERS
		test_data = [test_data(:, 1:11), test_data(:, 13:end)];
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
	rand("seed", 6);
	r = randperm(N_train);
	disp(r(1:10));
	train_data = train_data(r, :);

	%% There is no need to shuffle test data
end

disp(train_data(1:5, :));
pause

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






%% LINEAR REGRESSION

X_tr = [ones(N_train, 1) , X_tr]; %% Add the intercept


[theta, ~, ~, ~, results] = regress(y_tr, X_tr);

fprintf("training complete:\n");
fprintf(" theta: \n"); disp(theta');
fprintf("\n\n results: \n"); disp(results);
fprintf("\n");

predictions_normalized = [ones(N_test, 1) X_test] * theta;

fprintf("predictions  real values\n");
disp([predictions_normalized(1:10) y_test(1:10)]);


%% PLOTTING DATA
figure;

stem(X_tr(:,end), y_tr, "r", "marker", "x");
hold on;
stem(X_test(:,end), y_test, "b");
x = linspace(min(X_tr(:,12)), max(X_test(:,12)), 100);
plot(x, x*theta(end), 'g');

hold off;


predictions = predictions_normalized;
y = y_test;
if NORMALIZE_FEATURE
	predictions = mu_y + sigma_y * predictions_normalized;
	y = mu_y + sigma_y * y_test;
end

RMSE = sqrt( sum( (predictions - y).^2 ) / N_test );

RMSE





