function data =  get_all_data(base_dir, query, clear_outliers_flag=true)
	
	%% ======== READ DATA ===========

	sample = read_from_directory ([base_dir, query, "/small"]);
	big_sample = read_from_directory ([base_dir, query, "/big"]);

	%% ======== JOIN AND CLEAN OUTLIERS ===========

	complete_sample = [sample; big_sample];    %% Join data
	if clear_outliers_flag == true
		[clean_sample, indices] = clear_outliers (complete_sample);   %% Toglie i campioni anomali
	else
		clean_sample = complete_sample;
	end

	%% ======== SHUFFLE DATA ===========

	permutation = randperm (size (clean_sample, 1));   %% Ottiene una possibile permutazione di righe
	data = clean_sample(permutation, :);   %% Permuta le righe
end