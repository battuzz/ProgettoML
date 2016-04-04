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

## -*- texinfo -*- 
## @deftypefn {Function File} {@var{h} =} plot_R (@var{y}, @var{X})
##
## Plot response time, @var{y}, against number of cores and dataset size found
## in @var{X}.
## Return a handle @var{h} to the plot.
##
## @end deftypefn

function [h] = plot_R (y, X)

nCores = X(:, end);
dataSize = X(:, end-1);
h = figure;
plot3 (nCores, dataSize, y, "bx", "linewidth", 2);
grid on;
xlabel ('Number of cores');
ylabel ('Dataset size');
zlabel ('Response time');
title ('Job response time against cores and dataset size');

endfunction
