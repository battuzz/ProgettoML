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

query = "R5";
directory = ["/home/gianniti/policloud-runs/", query];

data = read_from_directory (directory);

values = data(:, 1);
sample = data(:, 2:end);

plot_R (values, sample);

figure_name = [directory, "/", query, "_3D.eps"];
print ("-depsc2", figure_name);
