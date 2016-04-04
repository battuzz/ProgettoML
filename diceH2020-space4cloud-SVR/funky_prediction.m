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
#base_dir = "/home/eugenio/Desktop/cineca-runs-20160116/";
base_dir = "/Users/Andrea/Documents/ProgettoML/cineca-runs-20160116/";

target = 10 * 60 * 1000; % In milliseconds
plot_minutes = false;
cluster = 40;
nUsers = 3;

C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);

train_frac = 0.8;

%% Real work
sample = read_from_directory ([base_dir, query]);

features = read_from_directory ([base_dir, query, "/predict"]);
labels = features(:, 1);
datasets = mod (labels, 1e4);
labels = floor (labels / 1e4);
features = features(:, 2:end);
features_nCores = features;
features_nCores(:, end) = 1 ./ features_nCores(:, end);

[clean_sample, ~] = clear_outliers (sample);
clean_sample_nCores = clean_sample;
clean_sample_nCores(:, end) = 1 ./ clean_sample_nCores(:, end);

rand ("seed", 17);
idx = randperm (size (clean_sample, 1));

shuffled = clean_sample(idx, :);
[scaled, mu, sigma] = zscore (shuffled);
y = scaled(:, 1);
X = scaled(:, 2:end);
mu_y = mu(1);
sigma_y = sigma(1);
mu_X = mu(2:end);
sigma_X = sigma(2:end);

shuffled_nCores = clean_sample_nCores(idx, :);
[scaled_nCores, mu, sigma] = zscore (shuffled_nCores);
y_nCores = scaled_nCores(:, 1);
X_nCores = scaled_nCores(:, 2:end);
mu_X_nCores = mu(2:end);
sigma_X_nCores = sigma(2:end);

test_frac = 1 - train_frac;
[ytr, ytst, ~] = split_sample (y, train_frac, test_frac);
[Xtr, Xtst, ~] = split_sample (X, train_frac, test_frac);
[ytr_nCores, ytst_nCores, ~] = split_sample (y_nCores, train_frac, test_frac);
[Xtr_nCores, Xtst_nCores, ~] = split_sample (X_nCores, train_frac, test_frac);

%% White box model, nCores
[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -q -p ", num2str(eps), " -c ", num2str(C)];
lm = svmtrain (ytr, Xtr, options);

%% White box model, nCores^(-1)
[C, eps] = model_selection (ytr_nCores, Xtr_nCores, ytst_nCores, Xtst_nCores, "-s 3 -t 0 -q -h 0", C_range, E_range);
options = ["-s 3 -t 0 -h 0 -q -p ", num2str(eps), " -c ", num2str(C)];
nlm = svmtrain (ytr_nCores, Xtr_nCores, options);

%% Predictions
safe_sigma_X = sigma_X + (sigma_X == 0);
safe_sigma_X_nCores = sigma_X_nCores + (sigma_X_nCores == 0);
scaled_features = bsxfun (@rdivide, bsxfun (@minus, features, mu_X), safe_sigma_X);
scaled_features_nCores = bsxfun (@rdivide, bsxfun (@minus, features_nCores, mu_X_nCores), safe_sigma_X_nCores);

scaled_predictions = svmpredict (labels, scaled_features, lm, "-q");
scaled_predictions_nCores = svmpredict (labels, scaled_features_nCores, nlm, "-q");

safe_sigma_y = sigma_y + (sigma_y == 0);
predictions = mu_y + scaled_predictions * sigma_y;
predictions_nCores = mu_y + scaled_predictions_nCores * sigma_y;

%% Obtain indices
R1_idx = (labels == 1);
R2_idx = (labels == 2);
R3_idx = (labels == 3);
R4_idx = (labels == 4);
R5_idx = (labels == 5);

d250_idx = (datasets == 250);
d500_idx = (datasets == 500);

%% Plot-related stuff
msec2min = 1;
if (plot_minutes)
  msec2min = 1 / (60 * 1000);
endif
target *= msec2min;

class1 = predictions(R1_idx & d250_idx) * msec2min;
class2 = predictions(R4_idx & d250_idx) * msec2min;

h1 = h2 = t1 = t2 = cell (1, nUsers);
for (r = 1:cluster - 1)
  s = cluster - r;
  for (users = 1:nUsers)
    div = 2 ^ (users - 1);
    time1 = class1(ceil (r / div));
    time2 = class2(ceil (s / div));
    found = false;
    if (time1 <= target)
      t1{users}(end+1) = time1;
      h1{users}(end+1) = r;
    endif
    if (time2 <= target)
      t2{users}(end+1) = time2;
      h2{users}(end+1) = r;
    endif
  endfor
endfor

function auxplot (x, y, idx, opt, name)
  dn = [name, " ", num2str(idx)];
  plot (x{idx}, y{idx}, opt, "linewidth", 2, "DisplayName", dn);
endfunction

crossplot = @(idx) auxplot (h1, t1, idx, "x", "R1 - Users");
ballplot = @(idx) auxplot (h2, t2, idx, "o", "R4 - Users");

figure;
hold all;
arrayfun (crossplot, 1:length (t1));
arrayfun (ballplot, 1:length (t2));
grid on;
legend location southwest;
