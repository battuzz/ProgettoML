function my_scatter(x, y, color='b', marker='o')

	handle = plot(x, y);
	set(handle, 'linestyle', 'none');
	set(handle, 'marker', marker);
	set(handle, 'color', color);
end