clear all;
clc;
close all hidden;

query = "everything/noMax";
base_dir = "/Users/Andrea/Documents/ProgettoML/cineca-runs-20160116/";

train_frac = 0.6;
test_frac = 0.2;

%% Prende i dati e li mischia
data = get_all_data(base_dir, query);

compl_time = data(:, 1);
n_map = data(:, 2);

data(:, end) = 1./data(:, end);

% data(:,end) = 1 ./ data(:,end);   %% Aggiunge la feature 1/nCore

[scaled, mu, sigma] = zscore(data);  %% normalizza

mu_y = mu(1);
sigma_y = sigma(1);

scaled(1:100, :)

y = scaled(:,1);
X = scaled(:, 2:end);

[theta, ~, ~, ~, ~] = regress(y, X);

predictions = X*theta;

y_rescaled = mu_y + y * sigma_y;
predictions_rescaled = mu_y + predictions * sigma_y;

output = [y_rescaled, predictions_rescaled, predictions_rescaled-y_rescaled, abs((predictions_rescaled - y_rescaled) ./ y_rescaled) ];

theta
mu
sigma

theta = mu(2:end) + sigma(2:end) .* theta';
theta = theta';

theta

for i = 2:size(data, 2)
	figure;
	stem(data(:, i), data(:, 1));
	hold on;
	x = linspace(1, max(data(:, i)), 1000);
	y = theta(i-1) * x;
	if i == 8
		plot(x/1000, y, 'r');
	else
		plot(x, y, 'r');	
	end
	hold off;
	pause
end



theta

