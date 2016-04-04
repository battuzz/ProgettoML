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
## @deftypefn {Function File} {@var{h} =} plot_MSE (@var{ytrain}, @var{Xtrain}, @var{ytest}, @var{Xtest}, @var{options}, @var{C}, @var{epsilon})
##
## Train an SVR model specified by @var{options} on the training set
## @var{ytrain}, @var{Xtrain} and plot the mean squared error obtained on the
## test set @var{ytest}, @var{Xtest}.
## All the combinations of values in @var{C} and @var{epsilon} are considered.
## Return the handle @var{h} to the plot.
##
## @end deftypefn

function h = plot_MSE (ytrain, Xtrain, ytest, Xtest, options, C, epsilon)

[cc, ee] = meshgrid (C, epsilon);
MSE = zeros (size (cc));
raw_options = options;

for (ii = 1:length (cc))
  options = [raw_options, " -q -c ", num2str(cc(ii)), " -p ", num2str(ee(ii))];
  model = svmtrain (ytrain, Xtrain, options);
  [~, accuracy, ~] = svmpredict (ytest, Xtest, model, "-q");
  MSE(ii) = accuracy(2);
endfor

h = surf (cc, ee, MSE);
xlabel ('C');
ylabel ('\epsilon');
zlabel ('MSE');
title ('MSE at varying model parameters');

endfunction
