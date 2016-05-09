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
## @deftypefn {Function File} {@var{h} =} plot_learning_curves (@var{m}, @var{MSE_train}, @var{MSE_cv})
##
## Plot learning curves given sample sizes @var{m} and mean squared errors,
## both on the training set @var{MSE_train} and on the cross validation
## set @var{MSE_cv}.
## Return the handle @var{h} to the plot.
##
## @seealso {learning_curves}
## @end deftypefn

function h = plot_learning_curves (m, MSE_train, MSE_cv, visible=true)

h = figure('Visible', visible);
plot (m, MSE_train, "b-", "linewidth", 2);
hold on;
plot (m, MSE_cv, "r-", "linewidth", 2);
legend ("Training set", "Test set");
xlabel ('m');
ylabel ('MSE');
title ('Learning curve at varying training set size');
grid on;
hold off;

endfunction
