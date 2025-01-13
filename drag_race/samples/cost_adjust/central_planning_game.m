function [player1_errors, player2_errors, player3_errors] = cost_adjustment(player1_games, player2_games, player3_games)

    numGames = length(player1_games);
    player1_errors = cell(numGames, 1);
    player2_errors = cell(numGames, 1);
    player3_errors = cell(numGames, 1);

    for i = 1:numGames
        A = player1_games{i};
        B = player2_games{i};
        C = player3_games{i};

        % Initial guess for errors (flattened)
        E_initial = zeros(3 * numel(A), 1);

        % Set optimization options
        options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'iter');

        % Minimize the objective function with constraints
        [E_opt, ~, exitflag] = fmincon(@(E)objective(E, A, B, C), ...
            E_initial, [], [], [], [], [], [], @(E)constraints(E, A, B, C), options);

        if exitflag ~= 1
            disp('Optimization did not converge');
        end

        % Extract optimized errors
        player1_errors{i} = reshape(E_opt(1:numel(A)), size(A));
        player2_errors{i} = reshape(E_opt(numel(A)+1:2*numel(B)), size(B));
        player3_errors{i} = reshape(E_opt(2*numel(B)+1:end), size(C));
    end
end

% Objective function (revised to include a penalty)
function f = objective(E, A, B, C)
    Ea = reshape(E(1:numel(A)), size(A));
    Eb = reshape(E(numel(A)+1:2*numel(B)), size(B));
    Ec = reshape(E(2*numel(B)+1:end), size(C));
    
    A_prime = A + Ea;
    B_prime = B + Eb;
    C_prime = C + Ec;
    phi = global_potential_function(A_prime, B_prime, C_prime);

    reg_term = 1e-6 * (norm(Ea(:)) + norm(Eb(:)) + norm(Ec(:)));
    
    % Add a penalty for potential values that approach zero (except phi(1,1,1))
    penalty_term = sum(max(0, 1e-3 - phi(:)));  % Penalize values too close to 0

    f = norm(phi(:)) + reg_term + penalty_term;
end

% Constraints function (now more strict on positivity)
function [c, ceq] = constraints(E, A, B, C)
    % Reshape E into error tensors for each player
    Ea = reshape(E(1:numel(A)), size(A));
    Eb = reshape(E(numel(A)+1:2*numel(B)), size(B));
    Ec = reshape(E(2*numel(B)+1:end), size(C));

    % Adjusted cost tensors
    A_prime = A + Ea;
    B_prime = B + Eb;
    C_prime = C + Ec;
    
    % Compute the global potential function
    phi = global_potential_function(A_prime, B_prime, C_prime);
    
    % Ensure all other values of phi are strictly greater than epsilon
    epsilon = 1e-3;  % Larger epsilon to enforce stricter positivity
    c = phi(2:end) - epsilon;  % All values except phi(1,1,1) must be > epsilon
    
    % Equality constraint to enforce phi(1,1,1) = 0
    ceq = phi(1) - 0;
end

% Global potential function (same as before)
function phi = global_potential_function(A, B, C)
    [m, n, p] = size(A);
    phi = zeros(m, n, p);
    phi(1,1,1) = 0;  % Base value

    % Fill the rest of phi using game dynamics
    for i = 2:m
        phi(i,1,1) = phi(i-1,1,1) + A(i,1,1) - A(i-1,1,1);
    end
    for j = 2:n
        phi(1,j,1) = phi(1,j-1,1) + B(1,j,1) - B(1,j-1,1);
    end
    for k = 2:p
        phi(1,1,k) = phi(1,1,k-1) + C(1,1,k) - C(1,1,k-1);
    end

    % Cross effects
    for i = 2:m
        for j = 2:n
            for k = 2:p
                phi(i,j,k) = (phi(i-1,j,k) + A(i,j,k) - A(i-1,j,k) + ...
                              phi(i,j-1,k) + B(i,j,k) - B(i,j-1,k) + ...
                              phi(i,j,k-1) + C(i,j,k) - C(i,j,k-1)) / 3;
            end
        end
    end
end

% Add errors to the original cost tensors
function [player1_adjusted, player2_adjusted, player3_adjusted] = add_errors(player1_errors, player2_errors, player3_errors, player1_games, player2_games, player3_games)
    numGames = length(player1_games);
    player1_adjusted = cell(numGames, 1);
    player2_adjusted = cell(numGames, 1);
    player3_adjusted = cell(numGames, 1);
    
    for i = 1:numGames
        player1_adjusted{i} = player1_games{i} + player1_errors{i};
        player2_adjusted{i} = player2_games{i} + player2_errors{i};
        player3_adjusted{i} = player3_games{i} + player3_errors{i};
    end
end

% Example usage
A1 = cat(3, [1 2; 4 5], [1 2; 3 4]);
A2 = cat(3, [3 4; 2 1], [4 2; 1 3]);

B1 = cat(3, [1 2; 4 5], [1 2; 3 4]);
B2 = cat(3, [1 5; 3 7], [3 7; 5 1]);

C1 = cat(3, [8 6; 4 2], [8 2; 6 4]);
C2 = cat(3, [2 8; 4 6], [4 6; 2 8]);

player1_games = {A1, A2};
player2_games = {B1, B2};
player3_games = {C1, C2};

% Compute error tensors
[player1_errors, player2_errors, player3_errors] = cost_adjustment(player1_games, player2_games, player3_games);

% Add errors to original costs
[player1_adjusted, player2_adjusted, player3_adjusted] = add_errors(player1_errors, player2_errors, player3_errors, player1_games, player2_games, player3_games);

% Compute global potential functions based on adjusted costs
potential_functions = cell(1, length(player1_adjusted));
for i = 1:length(player1_adjusted)
    potential = global_potential_function(player1_adjusted{i}, player2_adjusted{i}, player3_adjusted{i});
    potential_functions{i} = potential;
    
    % Display results
    fprintf('Subgame %d:\n', i);
    disp('Player 1 Error Tensor:');
    disp(player1_errors{i});
    disp('Player 2 Error Tensor:');
    disp(player2_errors{i});
    disp('Player 3 Error Tensor:');
    disp(player3_errors{i});
    disp('Global Potential Function:');
    disp(potential);
    disp(repmat('=', 1, 40));
end
