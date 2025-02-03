clear all

function [player2_errors, global_min_positions] = cost_adjustment(player1_games, player2_games, global_min_position)
    % Initialize lists to store error tensors and global minimum positions
    player2_errors = cell(size(player1_games));
    global_min_positions = cell(size(player1_games));
    
    for i = 1:length(player1_games)
        % Extract cost matrices for current game
        A = player1_games{i};
        B = player2_games{i};
        [m, n] = size(A);
        
        % Determine global minimum position for the game
        % [~, sec_policy1] = min(max(A, [], 2));
        % [~, sec_policy2] = max(min(A, [], 1));
        % global_min_position = [sec_policy1, sec_policy2];
        % global_min_positions{i} = global_min_position;
        % global_min_position = [2,2];
        global_min_positions{i} = global_min_position;
        
        % Compute initial potential function
        phi_initial = global_potential_function_numeric(A, B, global_min_position);
        
        if is_valid_exact_potential(A, B, phi_initial, global_min_position) ...
            && is_global_min_enforced(phi_initial, global_min_position)
            fprintf('Subgame %d: No adjustment needed.\n', i);
            player2_errors{i} = zeros(size(B)); % All zeros if no adjustment is needed
            continue;
        end
        
        % Use CVX to solve for Ea that minimizes the norm with constraints
        cvx_quiet true
        cvx_begin
            variable Ea(m, n) % Define Ea as an m x n matrix
            variable phi(m, n) % Define phi as an m x n potential function matrix
            minimize(norm(Ea, 'fro')) % Objective: minimize Frobenius norm of Ea
            
            % Constraint 1: Global minimum position at (0,0) for adjusted potential
            A_prime = A + Ea;
            phi(global_min_position(1), global_min_position(2)) == 0;
            
            % Constraint 2: Non-negativity for potential function
            epsilon = 1e-6;
            for k=1:m
                for j=1:n
                    if k ~= global_min_position(1) || j~= global_min_position(2)
                        phi(k,j) >= epsilon;
                    end
                end
            end
            
            % phi >= 0;

            % Constraint 3: Ensure exact potential condition
            for k = 2:m
                for l = 1:n
                    delta_A = A(k, l) - A(k - 1, l);
                    delta_phi = phi(k, l) - phi(k - 1, l);
                    delta_A == delta_phi;
                end
            end
            for k = 1:m
                for l = 2:n
                    delta_B = A_prime(k, l) - A_prime(k, l - 1);
                    delta_phi = phi(k, l) - phi(k, l - 1);
                    delta_B == delta_phi;
                end
            end
        cvx_end
        
        % Store the optimized Ea
        player2_errors{i} = Ea;
    end
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
A1 = [2, 2; 
      3, 4];
B1 = [4, 3; 
      2, 1];

A2 = [5, 2; 
    4, 5];
B2 = [1, 5; 
    3, 7];

A3 = [7, 1; 
    30, 0];
B3 = [8, 30; 
    0, 2];

A4 = [4, 5; 
    1, 3];
B4 = [1, 2; 
    4, 5];

% A = [0 0 0; -1 -1 -1; -2 -2 -2]
% A = [0 0 0; 0 0 0; 0 0 1]
% B = [0 -1 -2; 0 -1 -2; 0 -1 -2]

% A = [0 0 0 0 0; 
%      -1 -1 -1 -1 -1;
%      -2 -2 -2 -2 -2;
%      -3 -3 -3 -3 -3
%      -2 -2 -2 -2 -2;]
% B = [0 0 0 0 0; 
%      -1 -1 -1 -1 -1;
%      -2 -2 -2 -2 -2;
%      -3 -3 -3 -3 -3
%      -2 -2 -2 -1 -1;]'

% A = [0 0 0; 
%      -1 -1 -1;
%      0 0 0;]
% B = [0 -1 -2; 
%      0 -1 -2;
%      0 -1 -1;]

% A = [0 0 0; 
%      0 0 0;
%      0 0 0;]
% B = [0 -1 -2; 
%      0 -1 -2;
%      0 -1 -1;]

% A = [3 3 3; 
%      2 2 2;
%      1 1 1;]

% A = [0 1 2; 
%      -1 0 1;
%      -2 -1 0;]
% B = -A

% A = [0 0 -1; 0 0 -1; 1 1 0]
% B = -A

A = [0 2 2 1;-1 0 1 0; -2 0 0 -1; -1 1 1 0]'
B = -A

% List of test matrices for each subgame
% player1_games = {A1, A2, A3, A4};
% player2_games = {B1, B2, B3, B4};

player1_games = {A};
player2_games = {B};
% Compute potential functions for adjusted costs
potential_functions = cell(size(player2_games));
for i = 1:length(player2_games)
    % Run the cost adjustment function on test matrices
    for k = 1:size(A)
        for j = 1:size(A)

            global_min_position = [k, j];
            player1_errors = cost_adjustment(player1_games, player2_games, global_min_position);

            % Add errors to player 1's game matrix
            player1_adjusted = player1_games{i} + player1_errors{i};
            
            % Compute potential function for adjusted costs
            potential_functions{i} = global_potential_function_numeric(player1_adjusted, ...
                                                                       player2_games{i}, ...
                                                                       global_min_position);
                
            % Display test results for each subgame
            fprintf('Subgame %d Results:\n', i);
            disp('Player 1 Error:');
            disp(player1_errors{i});
            disp('Potential Function:');
            disp(potential_functions{i});
            disp('Global Min:');
            disp(global_min_position);
            disp(['Global Minimum Enforced: ', num2str( ...
                is_global_min_enforced( ...
                potential_functions{i},global_min_position) ...
                )]);
            disp(['Exact Potential: ', num2str(potential_functions{i}(global_min_position(1), global_min_position(2)) == 0)]);
            disp('----------------------------------------');
        end
    end
end
