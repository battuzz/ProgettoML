clear all;
clc;
close all hidden;

query = "everything/noMax";
base_dir = "/Users/Andrea/Documents/ProgettoML/cineca-runs-20160116/";

train_frac = 0.6;
test_frac = 0.2;

%% Prende i dati e li mischia
data = get_all_data(base_dir, query);

M = size(data, 2) - 1;   % numero di features
N = size(data, 1);   % numero di campioni


POTENZE = [1];
P = size(POTENZE, 2);


[data, ~] = clear_outliers(data);


N = size(data, 1);

% data(:,end) = 1 ./ data(:,end);   %% Aggiunge la feature 1/nCore

additional_features = ones(N, 1);

for i=1:M
	for j=1:P
		additional_features = [additional_features, data(:, i+1).^POTENZE(j)];
	end
end

data_with_features = [data(:, 1), additional_features(:,2:end)];



[scaled, mu, sigma] = zscore(data_with_features);  %% normalizza

mu_y = mu(1);
sigma_y = sigma(1);

mu_X = mu(2:end);
sigma_X = sigma(2:end);

y = scaled(:,1);
X = scaled(:, 2:end);







[theta, Bint, R, Rint, result] = regress(y, X);

theta
result


predictions_scaled = X*theta;

display("Ecco alcuni guess:\n    guess - valore reale");
display([predictions_scaled(1:10), y(1:10)]);


theta_unscaled = (mu_X + sigma_X .* theta')';


X_unscaled = zeros(size(X));
for i=1:(M*P)
	X_unscaled(:, i) = mu_X(i) .+ X(:,i) .* sigma_X(i);
end

y_unscaled = mu_y + sigma_y * y;


predictions_unscaled = mu_y + sigma_y * predictions_scaled;


% X_unscaled_without_feat = zeros(size(data(:, 2:end)));
% for i=1:M
% 	X_unscaled_without_feat(:, i) = X_unscaled(:, (i-1)*P+1)
% end



for i = 1:M
	figure;
	
	stem(X_unscaled(:, (i-1)*P+1), y_unscaled);
	hold on;
	x = linspace(min(X_unscaled(:,(i-1)*P+1)), max(X_unscaled(:, (i-1)*P+1)), 1000);

	y_values = zeros(size(x));
	
	for pow =1:P
		y_values = y_values + theta_unscaled((i-1)*P+pow) * x.^POTENZE(pow);
	end
	
	plot(x, y_values, 'r');	
	
	hold off;
	pause
end



% figure();
% stem(X_unscaled(:, 5), y_unscaled);
% hold on;
% x = linspace(0, 1000, 1000);
% y_values = theta_unscaled(5) * x;
% plot(x, y_values, 'r');
% hold off;


