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
## @deftypefn {Function File} {[@var{clean}, @var{indices}] =} clear_outliers (@var{dirty})
##
## Clear outliers from @var{dirty} by excluding rows where the value on a
## column is more than 3 standard deviations away from the mean.
## Return the @var{clean} dataset and the original @var{indices}
## kept after the procedure.
##
## @end deftypefn

function [clean, indices] = clear_outliers_ncores (dirty)

CORES_TO_SEARCH = [20,40,60,72,80,90,100,120];



cols = size(dirty, 2);     %% Numero di features

clean = dirty;
indices = 1:size(dirty, 1);  %% [1 2 3 4 5 6 7 .... N_righe]

for l = 1:length(CORES_TO_SEARCH)

	idx_cores = clean(:, end) == CORES_TO_SEARCH(l);		%% indexes of records with x cores
	idx_other = clean(:, end) != CORES_TO_SEARCH(l);

	if (sum(idx_cores) == 0) 
		continue;
	end
	
	good_idx = idx_cores;
	for (jj = 1:cols)
		avg = mean(clean(idx_cores, jj));	%% mean of completion times, considering only a single value of nCores
		dev = std(clean(idx_cores, jj));	%% standard deviation of "
		if (dev > 0)		%% Non consideriamo quelli con varianza 0 (esempio numero di core o dimensione tabella)
			good_idx = good_idx & (abs(clean(:, jj) - avg) < 2 * dev);	%% vettore con 1 se il test è buono, 0 se ha varianza > 2 * dev
		end
	end
	good_idx = good_idx | idx_other;
	
	clean = clean(good_idx, :);		%% Prende tutte le righe 'buone'
	indices = indices(good_idx);		%% Restituiremo anche gli indici delle righe che abbiamo tenuto
end

endfunction







