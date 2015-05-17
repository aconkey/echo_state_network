

%% Parameters =====================================================

N_in;					% number of input units
N_r;					% number of reservoir units
N_out;					% number of readout units

W_in;					% weights from input to reservoir
W_r;					% weights internal to reservoir
W_out;					% weights from reservoir to readout

scale_in				% scale factor of input weights
scale_r					% scale factor of reservoir weights

alpha;					% leaking rate of reservoir units
rho; 					% desired spectral radius of reservoir
density;				% sparsity coefficient for reservoir

seed = 1;				% random number generator seed
generator = 'twister' 	% type of random number generator

%==================================================================

% set random number generator:
rng(seed, generator);

% create sparse random reservoir:
temp_rand = 2 * scale_r * rand(N_r) - scale_r;
W_r = sprand(N_r, N_r, density);
W_r(find(W_r)) = temp_rand(find(W_r));
% *** NOTE *** Delete this at the end, temporary to work without sparse matrices:
W_r = full(W_r);

% scale reservoir weights to have desired spectral radius:
W_r = rho * (W_r ./ max(abs(eig(W_r))));

