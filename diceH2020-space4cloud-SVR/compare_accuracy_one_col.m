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
query = "R3_one_col";
dataSize = "500";
base_dir = "/home/eugenio/Desktop/cineca-runs-20160116/";

train_frac = 0.6;
test_frac = 0.2;

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);

printPlots = true;
scalePlots = true;
plot_subdivisions = 20;

%% Real stuff
sample = read_from_directory ([base_dir, query, "/", dataSize]);

%% Clear outliers grouping by number of cores
sample_before = sample;
sample = [];
cores = unique (sort (sample_before(:, end)))';
for (cr = cores)
  idx = (sample_before(:, end) == cr);
  [loc, ~] = clear_outliers (sample_before(idx, :));
  sample = [sample; loc];
endfor

%% Shuffle sample
rand ("seed", 17);
sample = sample(randperm (size (sample, 1)), :);
sample_nCores = sample;
sample_nCores(:, end) = 1 ./ sample_nCores(:, end);

values = sample(:, 1);
features = sample(:, 2:end);
[y, mu_y, sigma_y] = zscore (values);
[X, mu_X, sigma_X] = zscore (features);
[ytr, ytst, ycv] = split_sample (y, train_frac, test_frac);
[Xtr, Xtst, Xcv] = split_sample (X, train_frac, test_frac);

features_nCores = sample_nCores(:, 2:end);
[X_nCores, mu_X_nCores, sigma_X_nCores] = zscore (features_nCores);
[ytr_nCores, ytst_nCores, ycv_nCores] = split_sample (y, train_frac, test_frac);
[Xtr_nCores, Xtst_nCores, Xcv_nCores] = split_sample (X_nCores, train_frac, test_frac);

RMSEs = zeros (1, 4);
Cs = zeros (1, 4);
Es = zeros (1, 4);
predictions = zeros (numel (ycv), 4);
coefficients = cell (1, 4);
SVs = cell (1, 4);
b = cell (1, 4);
models = cell (1, 4);

%% White box model, nCores
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
models{1} = svmtrain (ytr, Xtr, options);
[predictions(:, 1), accuracy, ~] = svmpredict (ycv, Xcv, models{1});
Cs(1) = C;
Es(1) = eps;
RMSEs(1) = sqrt (accuracy(2));
coefficients{1} = models{1}.sv_coef;
SVs{1} = models{1}.SVs;
b{1} = - models{1}.rho;

%% White box model, nCores^(-1)
[C, eps] = model_selection (ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
models{2} = svmtrain (ytr_nCores, Xtr_nCores, options);
[predictions(:, 2), accuracy, ~] = svmpredict (ycv_nCores, Xcv_nCores, models{2});
Cs(2) = C;
Es(2) = eps;
RMSEs(2) = sqrt (accuracy(2));
coefficients{2} = models{2}.sv_coef;
SVs{2} = models{2}.SVs;
b{2} = - models{2}.rho;

%% Black box model, Polynomial
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 1 -q -h 0", C_range, E_range);
options = ["-s 3 -t 1 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
models{3} = svmtrain (ytr, Xtr, options);
[predictions(:, 3), accuracy, ~] = svmpredict (ycv, Xcv, models{3});
Cs(3) = C;
Es(3) = eps;
RMSEs(3) = sqrt (accuracy(2));
coefficients{3} = models{3}.sv_coef;
SVs{3} = models{3}.SVs;
b{3} = - models{3}.rho;

%% Black box model, RBF
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 2 -q -h 0", C_range, E_range);
options = ["-s 3 -t 2 -h 0 -p ", num2str(eps), " -c ", num2str(C)];
models{4} = svmtrain (ytr, Xtr, options);
[predictions(:, 4), accuracy, ~] = svmpredict (ycv, Xcv, models{4});
Cs(4) = C;
Es(4) = eps;
RMSEs(4) = sqrt (accuracy(2));
coefficients{4} = models{4}.sv_coef;
SVs{4} = models{4}.SVs;
b{4} = - models{4}.rho;

percent_RMSEs = 100 * RMSEs / max (RMSEs);

%% Unscale
real_predictions = mu_y + predictions * sigma_y;
cv_values = mu_y + ycv * sigma_y;

%% Compute metrics
abs_err = abs (real_predictions - cv_values);
rel_err = abs_err ./ cv_values;

max_rel_err = max (rel_err);
min_rel_err = min (rel_err);
mean_rel_err = mean (rel_err);

max_abs_err = max (abs_err);
mean_abs_err = mean (abs_err);
min_abs_err = min (abs_err);

mean_y = mean (cv_values);
mean_predictions = mean (real_predictions);
rel_err_mean = (mean_predictions - mean_y) / mean_y;

avgs = zeros (length (unique (sort (X))), 4);
err_on_avg = zeros (1, 4);
for (ii = 1:length (err_on_avg))
  dataset = X;
  if (ii == 2)
    dataset = X_nCores;
  endif
  cores = unique (sort (dataset));
  avg = zeros (size (cores));
  for (jj = 1:numel (cores))
    avg(jj) = mean (y(dataset == cores(jj)));
  endfor
  avgs(:, ii) = avg;
  [pred, ~, ~] = svmpredict (avg, cores, models{ii});
  pred = mu_y + pred * sigma_y;
  avg = mu_y + avg * sigma_y;
  err_on_avg(ii) = max (abs (pred - avg) ./ avg);
endfor

%% Plots
if (printPlots)
  figure;
  abscissae = X;
  ordinates = y;
  if (scalePlots)
    abscissae = features;
    ordinates = values;
  endif
  plot (abscissae, ordinates, "gx");
  hold on;
  cores = unique (sort (X));
  abscissae = cores;
  ordinates = avgs(:, 1);
  if (scalePlots)
    abscissae = mu_X + abscissae * sigma_X;
    ordinates = mu_y + ordinates * sigma_y;
  endif
  plot (abscissae, ordinates, "kd");
  w = SVs{1}' * coefficients{1};
  func = @(x) w .* x + b{1};
  M = max (X);
  m = min (X);
  abscissae = linspace (m, M, plot_subdivisions);
  ordinates = func (abscissae);
  if (scalePlots)
    abscissae = mu_X + abscissae * sigma_X;
    ordinates = mu_y + ordinates * sigma_y;
  endif
  plot (abscissae, ordinates, "r-", "linewidth", 2);
  axis auto;
  title ("Linear kernels");
  grid on;
  
  figure;
  abscissae = X_nCores;
  ordinates = y;
  if (scalePlots)
    abscissae = features_nCores;
    ordinates = values;
  endif
  plot (abscissae, ordinates, "gx");
  hold on;
  cores = unique (sort (X_nCores));
  abscissae = cores;
  ordinates = avgs(:, 2);
  if (scalePlots)
    abscissae = mu_X_nCores + abscissae * sigma_X_nCores;
    ordinates = mu_y + ordinates * sigma_y;
  endif
  plot (abscissae, ordinates, "kd");
  w = SVs{2}' * coefficients{2};
  func = @(x) w .* x + b{2};
  M = max (X_nCores);
  m = min (X_nCores);
  abscissae = linspace (m, M, plot_subdivisions);
  ordinates = func (abscissae);
  if (scalePlots)
    abscissae = mu_X_nCores + abscissae * sigma_X_nCores;
    ordinates = mu_y + ordinates * sigma_y;
  endif
  plot (abscissae, ordinates, "r-", "linewidth", 2);
  axis auto;
  title ('Linear kernels, nCores^{- 1}');
  grid on;
  
  figure;
  abscissae = X;
  ordinates = y;
  if (scalePlots)
    abscissae = features;
    ordinates = values;
  endif
  plot (abscissae, ordinates, "gx");
  hold on;
  cores = unique (sort (X));
  abscissae = cores;
  ordinates = avgs(:, 4);
  if (scalePlots)
    abscissae = mu_X + abscissae * sigma_X;
    ordinates = mu_y + ordinates * sigma_y;
  endif
  plot (abscissae, ordinates, "kd");
  M = max (X);
  m = min (X);
  abscissae = linspace (m, M, plot_subdivisions);
  ordinates = zeros (size (abscissae));
  for (ii = 1:numel (ordinates))
    point = abscissae(ii);
    ordinates(ii) = coefficients{4}' * exp (bsxfun (@minus, SVs{4}, point) .^ 2);
  endfor
  if (scalePlots)
    abscissae = mu_X + abscissae * sigma_X;
    ordinates = mu_y + ordinates * sigma_y;
  endif
  plot (abscissae, ordinates, "r-", "linewidth", 2);
  axis auto;
  title ("RBF kernels");
  grid on;
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

display ("Relative error between mean measure and mean prediction");
rel_err_mean

display ("Relative error between mean measure and mean prediction, grouped by number of cores");
err_on_avg
