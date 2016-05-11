addpath('./utility');

FOLDER = 'output/QUERY_COMP_CORE/SVR lineare/'
X_AXIS_LABEL = 'N Core'
Y_AXIS_LABEL = 'Completion time'

files_x = glob(strcat(FOLDER, '*_x.mat'));
files_y = glob(strcat(FOLDER, '*_y.mat'));

files_x = sort(files_x)
files_y = sort(files_y)

assert (length(files_x) == length(files_y));

figure()
hold on

legend_str = {}
color = {'r', 'b', 'g', 'k', 'c'};
for i = 1 : length(files_x)
	desc_x = strrep(files_x{i}, '_x.mat', '');
	desc_y = strrep(files_y{i}, '_y.mat', '');

	desc = strrep(desc_x, FOLDER, '');

	assert(strcmp(desc_x, desc_y));

	load(files_x{i}, 'x');
	load(files_y{i}, 'y');

	plot(x, y, color{i});

	legend_str{end + 1} = desc;

end

hold off;
legend(legend_str);
xlabel(X_AXIS_LABEL);
ylabel(Y_AXIS_LABEL);


