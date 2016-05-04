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
## @deftypefn {Function File} {[@var{training}, @var{testing}, @var{cv}] =} split_sample (@var{sample}, @var{train}, @var{test})
##
## Split @var{sample} so that a fraction @var{train}, @var{test}
## of the examples is, respectively, in the @var{training} set
## and in the @var{testing} set.
## The remaining examples are in the @var{cv} set.
##
## @end deftypefn

function [training, testing, cv] = split_sample (sample, train, test)

if (train + test > 1)
  error ("split_sample: wrong fractions");
endif

m = size (sample, 1);
m_train = round (m * train);
m_test = round (m * (test + train));
idx_train = 1:m_train;
idx_test = m_train+1:m_test;
idx_cv = m_test+1:m;

training = sample(idx_train, :);
testing = sample(idx_test, :);
cv = sample(idx_cv, :);

endfunction
