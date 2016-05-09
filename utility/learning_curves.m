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
## @deftypefn {Function File} {[@var{m}, @var{MSE_train}, @var{MSE_cv}] =} learning_curves (@var{ytrain}, @var{Xtrain}, @var{ycv}, @var{Xcv}, @var{options})
##
## Train an SVR model specified by @var{options} on the training set
## @var{ytrain}, @var{Xtrain} at varying dataset size.
## Return the arrays of sample sizes @var{m} and mean squared errors,
## both on the training set @var{MSE_train} and on the cross validation
## set @var{MSE_cv}.
##
## @seealso {plot_learning_curves}
## @end deftypefn

function [m, MSE_train, MSE_cv] = learning_curves (ytrain, Xtrain, ycv, Xcv, options)

m_train = length (ytrain);
m_cv = length (ycv);

m = round (linspace (m_cv, m_train, 20));
MSE_train = zeros (size (m));
MSE_cv = zeros (size (m));

for (ii = 1:length (m))
  m_part = m(ii);
  ytr = ytrain(1:m_part);
  Xtr = Xtrain(1:m_part, :);
  model = svmtrain (ytr, Xtr, options);
  [~, accuracy, ~] = svmpredict (ytr, Xtr, model, "-q");
  MSE_train(ii) = accuracy(2);
  [~, accuracy, ~] = svmpredict (ycv, Xcv, model, "-q");
  MSE_cv(ii) = accuracy(2);
endfor

endfunction
