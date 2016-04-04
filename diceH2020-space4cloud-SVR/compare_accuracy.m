## Copyright 2016 Eugenio Gianniti
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

clear all
close all hidden
clc

%% Parameters
query = "everything/noMax";
base_dir = "/Users/Andrea/Documents/ProgettoML/cineca-runs-20160116/";

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);

%% Train -> casi di training
%% Test -> casi usati per il model selection
%% Cross validation -> casi usati per il testing finale

train_frac = 0.6;
test_frac = 0.2;

printPlots = true;
scalePlots = true;
plot_subdivisions = 20;    %% 20

%% Real stuff

%% Legge i dati dalla directory

sample = read_from_directory ([base_dir, query, "/small"]);
big_sample = read_from_directory ([base_dir, query, "/big"]);

dims = size (sample);    %% Dimensione dei file 'small'
dimensions = dims(2) - 1;   %% Numero delle features
small_size = dims(1);   %% Numero di test case

%% Mette insieme i due set (big e small)

complete_sample = [sample; big_sample];
[clean_sample, indices] = clear_outliers (complete_sample);   %% Toglie i campioni anomali

clean_sample_nCores = clean_sample;
clean_sample_nCores(:, end) = 1 ./ clean_sample_nCores(:, end);  %% Aggiunge la feature 1/nCore

idx_small = (indices <= small_size);   %% Indici dei test SMALL 'buoni'
idx_big = (indices > small_size);     %% Indici dei test BIG 'buoni'

rand ("seed", 17);   %% Imposta il seed random
permutation = randperm (size (clean_sample, 1));   %% Ottiene una possibile permutazione di righe

[scaled, mu, sigma] = zscore (clean_sample);   %% Normalizza le feature

shuffled = scaled(permutation, :);   %% Permuta le righe

%% NOTA: Non vengono permutate le righe dei test case small e big singolarmente

y = scaled(idx_small, 1);
X = scaled(idx_small, 2:end);

big_y = scaled(idx_big, 1);
big_X = scaled(idx_big, 2:end);

all_y = shuffled(:, 1);
all_X = shuffled(:, 2:end);

%% Salva la media e varianza

mu_y = mu(1);
sigma_y = sigma(1);

mu_X = mu(2:end);
sigma_X = sigma(2:end);


[scaled_nCores, mu, sigma] = zscore (clean_sample_nCores);  %% Normalizza quella con la feature aggiuntiva 1/ncore
shuffled_nCores = scaled_nCores(permutation, :);

y_nCores = scaled_nCores(idx_small, 1);
X_nCores = scaled_nCores(idx_small, 2:end);

big_y_nCores = scaled_nCores(idx_big, 1);
big_X_nCores = scaled_nCores(idx_big, 2:end);

all_y_nCores = shuffled_nCores(:, 1);
all_X_nCores = shuffled_nCores(:, 2:end);

mu_X_nCores = mu(2:end);
sigma_X_nCores = sigma(2:end);





[ytr, ytst, ycv] = split_sample (all_y, train_frac, test_frac);
[Xtr, Xtst, Xcv] = split_sample (all_X, train_frac, test_frac);
[ytr_nCores, ytst_nCores, ycv_nCores] = split_sample (all_y_nCores, train_frac, test_frac);
[Xtr_nCores, Xtst_nCores, Xcv_nCores] = split_sample (all_X_nCores, train_frac, test_frac);

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


percent_RMSEs = 100 * RMSEs / max (RMSEs);     %% 1 -> RMSE peggiore. Più piccolo è meglio è.

%% Unscale
real_predictions = mu_y + predictions * sigma_y;   %% real_predictions: matrice N_casi * 4. Ogni colonna ha modello diverso
cv_values = mu_y + ycv * sigma_y;   %% Una sola colonna con i risultati misurati

%% Compute metrics
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

%% Plots
if (printPlots && (dimensions == 2))
  compare_accuracy_plot2D();
endif

%% Print metrics
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












function compare_accuracy_plot2D() 
  figure;
  XX = X(:, 1); 
  YY = X(:, 2);
  ZZ = y;
  if (scalePlots)
    XX = mu_X(1) + XX * sigma_X(1);
    YY = mu_X(2) + YY * sigma_X(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  plot3 (XX, YY, ZZ, "gx");
  hold on;
  XX = big_X(:, 1);
  YY = big_X(:, 2);
  ZZ = big_y;
  if (scalePlots)
    XX = mu_X(1) + XX * sigma_X(1);
    YY = mu_X(2) + YY * sigma_X(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  plot3 (XX, YY, ZZ, "bd");
  XX = Xcv(:, 1);
  YY = Xcv(:, 2);
  ZZ = ycv;
  if (scalePlots)
    XX = mu_X(1) + XX * sigma_X(1);
    YY = mu_X(2) + YY * sigma_X(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  plot3 (XX, YY, ZZ, "rx");
  w = SVs{1}' * coefficients{1};
  func = @(x, y) w(1) .* x + w(2) .* y + b{1};
  Ms = max ([X; big_X]);
  ms = min ([X; big_X]);
  x = linspace (ms(1), Ms(1), plot_subdivisions);
  yy = linspace (ms(2), Ms(2), plot_subdivisions);
  [XX, YY] = meshgrid (x, yy);
  ZZ = func (XX, YY);
  if (scalePlots)
    XX = mu_X(1) + XX * sigma_X(1);
    YY = mu_X(2) + YY * sigma_X(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  surf (XX, YY, ZZ);
  axis auto;
  title ("Linear kernels");
  grid on;
  
  figure;
  XX = X_nCores(:, 1);
  YY = X_nCores(:, 2);
  ZZ = y_nCores;
  if (scalePlots)
    XX = mu_X_nCores(1) + XX * sigma_X_nCores(1);
    YY = mu_X_nCores(2) + YY * sigma_X_nCores(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  plot3 (XX, YY, ZZ, "gx");
  hold on;
  XX = big_X_nCores(:, 1);
  YY = big_X_nCores(:, 2);
  ZZ = big_y_nCores;
  if (scalePlots)
    XX = mu_X_nCores(1) + XX * sigma_X_nCores(1);
    YY = mu_X_nCores(2) + YY * sigma_X_nCores(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  plot3 (XX, YY, ZZ, "bd");
  XX = Xcv_nCores(:, 1);
  YY = Xcv_nCores(:, 2);
  ZZ = ycv_nCores;
  if (scalePlots)
    XX = mu_X_nCores(1) + XX * sigma_X_nCores(1);
    YY = mu_X_nCores(2) + YY * sigma_X_nCores(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  plot3 (XX, YY, ZZ, "rx");
  w = SVs{2}' * coefficients{2};
  func = @(x, y) w(1) .* x + w(2) .* y + b{2};
  Ms = max ([X_nCores; big_X_nCores]);
  ms = min ([X_nCores; big_X_nCores]);
  x = linspace (ms(1), Ms(1), plot_subdivisions);
  yy = linspace (ms(2), Ms(2), plot_subdivisions);
  [XX, YY] = meshgrid (x, yy);
  ZZ = func (XX, YY);
  if (scalePlots)
    XX = mu_X_nCores(1) + XX * sigma_X_nCores(1);
    YY = mu_X_nCores(2) + YY * sigma_X_nCores(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  surf (XX, YY, ZZ);
  axis auto;
  title ('Linear kernels, nCores^{- 1}');
  grid on;
  
  figure;
  XX = X(:, 1);
  YY = X(:, 2);
  ZZ = y;
  if (scalePlots)
    XX = mu_X(1) + XX * sigma_X(1);
    YY = mu_X(2) + YY * sigma_X(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  plot3 (XX, YY, ZZ, "gx");
  hold on;
  XX = big_X(:, 1);
  YY = big_X(:, 2);
  ZZ = big_y;
  if (scalePlots)
    XX = mu_X(1) + XX * sigma_X(1);
    YY = mu_X(2) + YY * sigma_X(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  plot3 (XX, YY, ZZ, "bd");
  XX = Xcv(:, 1);
  YY = Xcv(:, 2);
  ZZ = ycv;
  if (scalePlots)
    XX = mu_X(1) + XX * sigma_X(1);
    YY = mu_X(2) + YY * sigma_X(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  plot3 (XX, YY, ZZ, "rx");
  Ms = max ([X; big_X]);
  ms = min ([X; big_X]);
  x = linspace (ms(1), Ms(1), plot_subdivisions);
  yy = linspace (ms(2), Ms(2), plot_subdivisions);
  [XX, YY] = meshgrid (x, yy);
  [nr, nc] = size (XX);
  ZZ = zeros (nr, nc);
  for (r = 1:nr)
    for (c = 1:nc)
      point = [XX(r, c), YY(r, c)];
      ZZ(r, c) = coefficients{4}' * exp (sumsq (bsxfun (@minus, SVs{4}, point), 2) / 2);
    endfor
  endfor
  if (scalePlots)
    XX = mu_X(1) + XX * sigma_X(1);
    YY = mu_X(2) + YY * sigma_X(2);
    ZZ = mu_y + ZZ * sigma_y;
  endif
  surf (XX, YY, ZZ);
  axis auto;
  title ("RBF kernels");
  grid on;

end;
