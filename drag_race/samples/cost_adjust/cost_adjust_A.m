clear all
clc

function error = cost_adjustment(A, B, min_position)
    [m, n] = size(A);
    error = [];
    % Compute initial potential function
    phi_initial = global_potential_function_numeric(A, B, min_position);

    % Use CVX to solve for Ea that minimizes the norm with constraints
    cvx_quiet true
    cvx_begin
        variable E(m, n) % Define Ea as an m x n matrix
        variable phi(m, n) % Define phi as an m x n potential function matrix
        minimize(norm(E, 'fro')) % Objective: minimize Frobenius norm of Ea
        
        % Constraint 1: Global minimum position at (0,0) for adjusted potential
        A_prime = A + E;
        phi(min_position(1), min_position(2)) == 0;
        
        % Constraint 2: Non-negativity for potential function
        epsilon = 1e-6;
        for k=1:m
            for j=1:n
                if k ~= min_position(1) || j~= min_position(2)
                    phi(k,j) >= epsilon;
                end
            end
        end
        
        % Constraint 3: Ensure exact potential condition
        for k = 2:m
            for l = 1:n
                delta_A = A_prime(k-1, l) - A_prime(k, l);
                delta_phi = phi(k-1, l) - phi(k, l);
                delta_A == delta_phi;
            end
        end
        for k = 1:m
            for l = 2:n
                delta_B = B(k, l) - B(k, l - 1);
                delta_phi = phi(k, l) - phi(k, l - 1);
                delta_B == delta_phi;
            end
        end
    cvx_end
    error = E;
end

function phi = global_potential_function_numeric(A, B, global_min_position)
    % Computes a global potential function for players given cost matrices A and B
    [m, n] = size(A);
    phi = zeros(m, n);
    for i = 2:m
        phi(i, 1) = phi(i - 1, 1) + A(i, 1) - A(i - 1, 1);
    end
    for j = 2:n
        phi(1, j) = phi(1, j - 1) + B(1, j) - B(1, j - 1);
    end
    for i = 2:m
        for j = 2:n
            phi(i, j) = (phi(i - 1, j) + A(i, j) - A(i - 1, j) + phi(i, j - 1) + B(i, j) - B(i, j - 1)) / 2;
        end
    end
    phi = phi - phi(global_min_position(1), global_min_position(2));
end

function valid = is_valid_exact_potential(A, B, phi, global_min_position)
    % Validate exact potential condition for A and B
    valid = true;
    [m, n] = size(A);
    for i = 2:m
        for j = 1:n
            if abs((A(i, j) - A(i - 1, j)) - (phi(i, j) - phi(i - 1, j))) > 1e-6
                valid = false;
                return;
            end
        end
    end
    for i = 1:m
        for j = 2:n
            if abs((B(i, j) - B(i, j - 1)) - (phi(i, j) - phi(i, j - 1))) > 1e-6
                valid = false;
                return;
            end
        end
    end
end

function valid = is_global_min_enforced(phi, global_min_position)
    
    [m, n] = size(phi);  % Get the dimensions of the matrix
    valid = phi(global_min_position(1), global_min_position(2)) == 0;

    for i=1:m
        for j=1:n
            if i ~= global_min_position(1) || j~= global_min_position(2)
                valid = valid && (phi(i,j) > 0);
            end
        end
    end
end

% Define test cost matrices for Player 1 and Player 2
A1 = [0 1 2; 
    -1 0 1; 
    -2 -1 0];

A2 = [0 1 2;
      1 2 3;
      2 3 4];

B1 = -A1;
B2 = A2;

B = B1 + 2*B2;

[a, p2_sec] = min(max(B,[],1));

for i = 1:length(A1)-1
    min_position = [i, p2_sec];
    E = cost_adjustment(A1, B, min_position);
    if isempty(E)
        continue;
    end

    A_prime = A1 + E;
    phi = global_potential_function_numeric(A_prime, B, min_position);

    if is_valid_exact_potential(A_prime, B, phi, min_position) && is_global_min_enforced(phi, min_position)

        fprintf('Subgame %d Results:\n', i);
        disp('Player 1 Error:');
        disp(E);
        disp('A_prime:');
        disp(A_prime);
        disp('B:');
        disp(B);
        disp('Potential Function:');
        disp(phi);
        disp('Global Min:');
        disp(min_position);
        disp(['Global Minimum Enforced: ', num2str(is_global_min_enforced(phi,min_position))]);
        disp(['Exact Potential: ', num2str(is_valid_exact_potential(A_prime, B, phi, min_position))]);
        disp('----------------------------------------');
    end
end
