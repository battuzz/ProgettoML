clear all;
clc;
close all hidden;

%% ======== VARIABLES ===========

query = "everything/noMax";
base_dir = "/Users/Andrea/Documents/ProgettoML/cineca-runs-20160116/";

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);

train_frac = 0.6;
test_frac = 0.2;


%% ======== READ DATA ===========
display("Reading data from files\n");
fflush(stdout);

sample = read_from_directory ([base_dir, query, "/small"]);
big_sample = read_from_directory ([base_dir, query, "/big"]);

%% ======== JOIN AND CLEAN OUTLIERS ===========
display("Joining and cleaning data");
fflush(stdout);

complete_sample = [sample; big_sample];    %% Join data
[clean_sample, indices] = clear_outliers (complete_sample);   %% Toglie i campioni anomali

%% ======== ADD NEW FEATURES ===========
display("Adding new features");
fflush(stdout);

clean_sample_nCores = clean_sample;
clean_sample_nCores(:, end) = 1 ./ clean_sample_nCores(:, end);  %% Aggiunge la feature 1/nCore

%% ======== NORMALIZE ===========
display("Normalizing");
fflush(stdout);

[scaled, mu, sigma] = zscore (clean_sample);   %% Normalizza le feature
[scaled_nCores, mu_nCores, sigma_nCores] = zscore (clean_sample_nCores);   %% Normalizza le feature

mu_y = mu(:, 1);
sigma_y = sigma(:, 1);

%% ======== SHUFFLE DATA ===========
display("Shuffling data");
fflush(stdout);

permutation = randperm (size (scaled, 1));   %% Ottiene una possibile permutazione di righe
scaled = scaled(permutation, :);   %% Permuta le righe

permutation = randperm (size (scaled_nCores, 1));   %% Ottiene una possibile permutazione di righe
scaled_nCores = scaled_nCores(permutation, :);   %% Permuta le righe



%% ======== SPLIT SAMPLES ===========
display("Splitting data for model selection");
fflush(stdout);

y = scaled(:,1);  % Prende la prima colonna
X = scaled(:, 2:end); % Prende tutto il resto

y_nCores = scaled_nCores(:,1);  % Prende la prima colonna
X_nCores = scaled_nCores(:, 2:end); % Prende tutto il resto


[ytr, ytst, ycv] = split_sample (y, train_frac, test_frac);
[Xtr, Xtst, Xcv] = split_sample (X, train_frac, test_frac);

[ytr_nCores, ytst_nCores, ycv_nCores] = split_sample (y_nCores, train_frac, test_frac);
[Xtr_nCores, Xtst_nCores, Xcv_nCores] = split_sample (X_nCores, train_frac, test_frac);




%% ======== MODEL SELECTION & TRAIN ===========
display("Begin with model selection");
fflush(stdout);

RMSEs = zeros (1, 4);
Cs = zeros (1, 4);     
Es = zeros (1, 4);
predictions = zeros (numel (ycv), 4);
coefficients = cell (1, 4);
SVs = cell (1, 4);
b = cell (1, 4);

% Parametri per svmtrain
% -s --> tipo di SVM (3 = epsilon-SVR)
% -t --> tipo di kernel (0 = lineare, 1 = polinomiale, 2 = gaussiano, 3 = sigmoide)
% -q --> No output
% -h --> (0 = No shrink)
% -p --> epsilon
% -c --> cost


display("\nWhite box linear without additional features");
fflush(stdout);
%% White box model, nCores  LINEAR
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[predictions(:, 1), accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(1) = C;
Es(1) = eps;
RMSEs(1) = sqrt (accuracy(2));
coefficients{1} = model.sv_coef;
SVs{1} = model.SVs;
b{1} = - model.rho;

display("\nWhite box linear with additional features");
fflush(stdout);
%% White box model, nCores^(-1)   LINEAR
[C, eps] = model_selection (ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr_nCores, Xtr_nCores, options);
[predictions(:, 2), accuracy, ~] = svmpredict (ycv_nCores, Xcv_nCores, model);
Cs(2) = C;
Es(2) = eps;
RMSEs(2) = sqrt (accuracy(2));
coefficients{2} = model.sv_coef;
SVs{2} = model.SVs;
b{2} = - model.rho;

display("\nBlack box polynomial without additional features");
fflush(stdout);
%% Black box model, Polynomial
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 1 -q -h 0", C_range, E_range);
options = ["-s 3 -t 1 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[predictions(:, 3), accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(3) = C;
Es(3) = eps;
RMSEs(3) = sqrt (accuracy(2));
coefficients{3} = model.sv_coef;
SVs{3} = model.SVs;
b{3} = - model.rho;

display("\nBlack box RBF without additional features");
fflush(stdout);
%% Black box model, RBF (Radial Basis Function)
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 2 -q -h 0", C_range, E_range);
options = ["-s 3 -t 2 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
model = svmtrain (ytr, Xtr, options);
[predictions(:, 4), accuracy, ~] = svmpredict (ycv, Xcv, model);
Cs(4) = C;
Es(4) = eps;
RMSEs(4) = sqrt (accuracy(2));
coefficients{4} = model.sv_coef;
SVs{4} = model.SVs;
b{4} = - model.rho;







%% ======== UNSCALE PREDICTIONS ===========
display("Unscaling data");
fflush(stdout);

real_predictions = mu_y + predictions * sigma_y;

cv_values = mu_y + ycv * sigma_y;



%% ======== COMPUTE METRICS ===========
display("Computing metrics");
fflush(stdout);
percent_RMSEs = 100 * RMSEs / max (RMSEs);     %% 1 -> RMSE peggiore. Più piccolo è meglio è.

abs_err = abs (real_predictions - cv_values);
rel_err = abs_err ./ cv_values;

max_rel_err = max (rel_err);
min_rel_err = min (rel_err);
mean_rel_err = mean (rel_err);

max_abs_err = max (abs_err);
mean_abs_err = mean (abs_err);
min_abs_err = min (abs_err);

mean_values = mean (cv_values);
mean_predictions = mean (real_predictions);
err_mean = mean_predictions - mean_values;
rel_err_mean = err_mean / mean_values;


%% ======== PRINT RESULTS ===========
display("Run finished: printing results: ");
fflush(stdout);

display ("Root Mean Square Errors");
RMSEs
percent_RMSEs

display ("Relative errors (absolute values)");
max_rel_err
mean_rel_err
min_rel_err

display ("Absolute errors (absolute values)");
max_abs_err
mean_abs_err
min_abs_err

display ("Relative error between mean measure and mean prediction (absolute value)");
rel_err_mean















