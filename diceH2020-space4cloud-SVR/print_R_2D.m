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
directory = ["/Users/Andrea/Documents/ProgettoML/cineca-runs-20160116/", query];

sample = read_from_directory (directory);

y = sample(:, 1);
X = sample(:, 2:end);

dataSize = X(:, end-1);
sizes = unique (sort (dataSize))';

nCores = X(:, end);

for (sz = sizes)

  idx = (dataSize == sz);
  nCores_loc = nCores(idx);
  y_loc = y(idx);

  figure;
  plot (nCores_loc, y_loc, "bx", "linewidth", 2);
  grid on;
  xlabel ('Number of cores');
  ylabel ('Response time');
  size_string = num2str (sz);
  title_string = ['Job response time against cores at ', size_string, ' GB'];
  title (title_string);

  values = unique (nCores_loc);
  avg = zeros (size (values));
  for (ii = 1:length (values))
    avg(ii) = mean (y_loc(nCores_loc == values(ii)));
  endfor
  hold on;
  plot (values, avg, "r:", "linewidth", 2);
  hold off;

  figure_name = [directory, "/", query, "_s", size_string, ".eps"];
  print ("-depsc2", figure_name);

  close all hidden;

endfor
