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
% query = "unknown/max";
query = "everything/noMax";
%%base_dir = "/Users/Andrea/Documents/ProgettoML/cineca-runs-20160116/"
base_dir = "/Users/Andrea/Documents/ProgettoML/cineca-runs-20160116/"
seeds = 1:17;

train_frac = 0.6;
test_frac = 0.2;

%% Model stuff
C_range = linspace (0.1, 5, 20);
E_range = linspace (0.1, 5, 20);
initial_options = "-s 3 -t 0 -q";

%% Work
sample = read_from_directory ([base_dir, query, "/small"]);
sample_big = read_from_directory ([base_dir, query, "/big"]);

small_size = size (sample, 1);
complete_sample = [sample; sample_big];
small_idx = (1:size (complete_sample, 1) <= small_size);
scaled = zscore (complete_sample);

small_scaled = scaled(small_idx, :);
big_scaled = scaled(!small_idx, :);

rand ("seed", seeds(1));
shuffled = scaled(randperm (size (small_scaled, 1)), :);
[train, test, ~] = split_sample (shuffled, train_frac, test_frac);
ytr = train(:, 1);
Xtr = train(:, 2:end);
ytst = test(:, 1);
Xtst = test(:, 2:end);

[C, eps] = model_selection (ytr, Xtr, ytst, Xtst, initial_options, C_range, E_range);
options = [initial_options, " -p ", num2str(eps), " -c ", num2str(C)];

seeds = seeds(:)';
m = MSEtrain = MSEcv = [];
for (seed = seeds)
  rand ("seed", seed);
  idx = randperm (size (small_scaled, 1));
  shuffled = small_scaled(idx, :);
  [train, ~, ~] = split_sample (shuffled, train_frac, test_frac);
  ytr = train(:, 1);
  Xtr = train(:, 2:end);
  ycv = big_scaled(:, 1);
  Xcv = big_scaled(:, 2:end);
  
  [current_m, current_MSEtrain, current_MSEcv] = learning_curves (ytr, Xtr, ycv, Xcv, options);
  m = [m; current_m];
  MSEtrain = [MSEtrain; current_MSEtrain];
  MSEcv = [MSEcv; current_MSEcv];
endfor

old_m = m;
m = mean (old_m);
if (any (m != old_m))
  error ("Something went wrong with the sample size steps");
endif
MSEtrain = mean (MSEtrain);
MSEcv = mean (MSEcv);

plot_learning_curves (m, MSEtrain, MSEcv);
