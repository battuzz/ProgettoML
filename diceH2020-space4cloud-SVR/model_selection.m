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
## @deftypefn {Function File} {[@var{C}, @var{epsilon}] =} model_selection (@var{ytrain}, @var{Xtrain}, @var{ytest}, @var{Xtest}, @var{options}, @var{C_range}, @var{epsilon_range})
##
## Perform model selection on the training set @var{ytrain}, @var{Xtrain}
## according to the performance on the test set @var{ytest], @var{Xtest}.
## The general model is defined via @var{options} and the grid search spans
## @var{C_range} and @var{epsilon_range}.
## In the end, return the optimal @var{C} and @var{epsilon}.
##
## @end deftypefn

## Author:  <eugenio@archlinuxVM>
## Created: 2015-11-04

function [C, epsilon] = model_selection (ytrain, Xtrain, ytest, Xtest, options, C_range, epsilon_range)

raw_options = options;
C_range = C_range(:)';
epsilon_range = epsilon_range(:)';

C = Inf;
epsilon = Inf;
MSE = Inf;
for (cc = C_range)
  for (ee = epsilon_range)
    options = [raw_options, " -p ", num2str(ee), " -c ", num2str(cc)];
    model = svmtrain (ytrain, Xtrain, options);
    [~, accuracy, ~] = svmpredict (ytest, Xtest, model, "-q");
    mse = accuracy(2);
    if (mse < MSE)
      C = cc;
      epsilon = ee;
      MSE = mse;
    endif
  endfor
endfor

endfunction
