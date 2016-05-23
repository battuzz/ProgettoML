function train_data = get_all_data_from_dirs(base, queries)
	train_data = [];
	for q = queries
		fprintf("\nDEBUG: loading %s\n", char(q));
		train_data = [train_data; read_from_directory(strcat(char(base), char(q)))];
	endfor
endfunction
