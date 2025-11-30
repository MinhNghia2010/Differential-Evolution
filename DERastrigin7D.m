% -------------------------------------------------------------------------
% Author: Nnaji Obinna
% Student ID: 101001461
% Institution: Ontario Tech University
% Project: Rastrigin Function Optimization using GA, DE, and PSO
% Description: DE variants comparison on 7D Rastrigin function
% Copyright (c) 2025 Nnaji Obinna & Ontario Tech University.
% All rights reserved.
% -------------------------------------------------------------------------
function DERastrigin7D()
clc; clear; close all;

fprintf('=== RUNNING DE ON 7D RASTRIGIN FUNCTION ===\n');

%% DE Parameters for 7D
pop_size = 70;
num_gen = 300;
dim = 7;
lb = -5.12;
ub = 5.12;

% Strategies to compare
mutation_strategies = {'DE/rand/1 (Original)', 'DE/best/1', 'DE/rand/2', 'DE/current-to-best/1', 'Adaptive DE'};
best_curves = cell(5, 1);
best_solutions = cell(5, 1);
best_fitnesses = zeros(5, 1);
computation_times = zeros(5, 1);

%% Run all 5 DE variants
for strategy = 1:5
    fprintf('Running %s for 7D Rastrigin...\n', mutation_strategies{strategy});
    tic;
    
    %% Initialize Population
    pop = lb + (ub - lb) * rand(pop_size, dim);
    
    % Evaluate fitness
    fitness = zeros(pop_size, 1);
    for i = 1:pop_size
        fitness(i) = rastrigin_7d(pop(i, :));
    end

    [best_fitness, idx] = min(fitness);
    best_solution = pop(idx, :);
    best_curve = zeros(num_gen, 1);
    
    % Initialize variables for adaptive DE
    if strategy == 5
        success_history = [];
        F_history = [];
        CR_history = [];
        strategy_performance = ones(4, 10);
        current_strategy = 1;
    end

    %% DE Main Loop
    for gen = 1:num_gen
        [~, global_best_idx] = min(fitness);
        successful_mutations = 0;
        
        for i = 1:pop_size
            %% ADAPTIVE DE: Auto-select parameters and strategy
            if strategy == 5
                [F, CR, current_strategy] = adaptive_parameters(...
                    gen, success_history, F_history, CR_history, strategy_performance);
            else
                % Fixed parameters for non-adaptive strategies
                F = 0.8;
                CR = 0.9;
            end
            
            % Mutation: Select strategy
            switch strategy
                case 1 % DE/rand/1 (Original)
                    idxs = randperm(pop_size, 3);
                    while any(idxs == i)
                        idxs = randperm(pop_size, 3);
                    end
                    a = pop(idxs(1), :);
                    b = pop(idxs(2), :);
                    c = pop(idxs(3), :);
                    mutant = a + F * (b - c);
                    
                case 2 % DE/best/1
                    [~, best_idx] = min(fitness);
                    idxs = randperm(pop_size, 2);
                    while any(idxs == i) || any(idxs == best_idx)
                        idxs = randperm(pop_size, 2);
                    end
                    a = pop(best_idx, :);
                    b = pop(idxs(1), :);
                    c = pop(idxs(2), :);
                    mutant = a + F * (b - c);
                    
                case 3 % DE/rand/2
                    idxs = randperm(pop_size, 5);
                    while any(idxs == i)
                        idxs = randperm(pop_size, 5);
                    end
                    a = pop(idxs(1), :);
                    b = pop(idxs(2), :);
                    c = pop(idxs(3), :);
                    d = pop(idxs(4), :);
                    e = pop(idxs(5), :);
                    mutant = a + F * (b - c) + F * (d - e);
                    
                case 4 % DE/current-to-best/1
                    idxs = randperm(pop_size, 2);
                    while any(idxs == i) || any(idxs == global_best_idx)
                        idxs = randperm(pop_size, 2);
                    end
                    current = pop(i, :);
                    best = pop(global_best_idx, :);
                    r1 = pop(idxs(1), :);
                    r2 = pop(idxs(2), :);
                    mutant = current + F * (best - current) + F * (r1 - r2);
                    
                case 5 % Adaptive DE
                    mutant = apply_strategy(current_strategy, pop, i, F, global_best_idx, fitness);
            end
            
            mutant = max(min(mutant, ub), lb); % Bound check

            % Crossover
            trial = pop(i, :);
            j_rand = randi(dim);
            for j = 1:dim
                if rand < CR || j == j_rand
                    trial(j) = mutant(j);
                end
            end

            % Selection
            trial_fit = rastrigin_7d(trial);
            
            if trial_fit < fitness(i)
                pop(i, :) = trial;
                fitness(i) = trial_fit;
                successful_mutations = successful_mutations + 1;
                
                % Update adaptive DE information
                if strategy == 5
                    F_history = [F_history, F];
                    CR_history = [CR_history, CR];
                    strategy_performance(current_strategy, mod(gen-1, 10)+1) = ...
                        strategy_performance(current_strategy, mod(gen-1, 10)+1) + 1;
                end
            end
        end

        % Update success history for adaptive DE
        if strategy == 5
            success_rate = successful_mutations / pop_size;
            success_history = [success_history, success_rate];
            
            % Keep history size reasonable
            if length(success_history) > 20
                success_history = success_history(end-19:end);
            end
            if length(F_history) > 50
                F_history = F_history(end-49:end);
            end
            if length(CR_history) > 50
                CR_history = CR_history(end-49:end);
            end
        end

        % Track best solution
        [current_best, idx] = min(fitness);
        if current_best < best_fitness
            best_fitness = current_best;
            best_solution = pop(idx, :);
        end

        best_curve(gen) = best_fitness;
        
        % Display progress for adaptive DE
        if strategy == 5 && mod(gen, 50) == 0
            fprintf('  Generation %d: Best Fitness = %.4f\n', gen, best_fitness);
        end
    end

    computation_times(strategy) = toc;
    
    % Store results
    best_curves{strategy} = best_curve;
    best_solutions{strategy} = best_solution;
    best_fitnesses(strategy) = best_fitness;
    
    % Display results
    fprintf('%s - Best Fitness: %.4f, Time: %.2fs\n\n', ...
            mutation_strategies{strategy}, best_fitness, computation_times(strategy));
end

%% Plot results
figure;
colors = ['r', 'b', 'g', 'm', 'y'];
line_styles = {'-', '--', ':', '-.', '-'};
line_widths = [1.5, 1.5, 1.5, 1.5, 2.5];

hold on;
for i = 1:5
    plot(1:num_gen, best_curves{i}, 'Color', colors(i), 'LineStyle', line_styles{i}, ...
         'LineWidth', line_widths(i), 'DisplayName', mutation_strategies{i});
end
hold off;

xlabel('Generation');
ylabel('Best Fitness');
title('DE Convergence Comparison on 7D Rastrigin Function');
legend('show', 'Location', 'northeast');
grid on;
set(gca, 'YScale', 'log');

%% Display summary results
fprintf('\n=== FINAL RESULTS FOR 7D RASTRIGIN ===\n');
fprintf('%-25s %-12s %-10s\n', 'Strategy', 'Best Fitness', 'Time (s)');
fprintf('%-25s %-12s %-10s\n', '-----------------------', '-----------', '--------');
for i = 1:5
    fprintf('%-25s %-12.4f %-10.2f\n', mutation_strategies{i}, best_fitnesses(i), computation_times(i));
end

[best_overall_fitness, best_idx] = min(best_fitnesses);
fprintf('\nBEST PERFORMER: %s with fitness = %.6f\n', ...
        mutation_strategies{best_idx}, best_overall_fitness);

end

%% Adaptive DE Functions
function [F, CR, strategy_idx] = adaptive_parameters(gen, success_history, F_history, CR_history, strategy_performance)
    % Adaptive parameter and strategy selection
    
    if isempty(success_history)
        % Random initialization
        F = 0.5 + rand() * 0.5;
        CR = 0.5 + rand() * 0.4;
        strategy_idx = randi(4);
    else
        % Adapt F based on success history
        if ~isempty(F_history)
            recent_success = mean(success_history(max(1, end-4):end));
            if recent_success > 0.1
                F = 0.6 + rand() * 0.4;
            else
                F = 0.3 + rand() * 0.4;
            end
        else
            F = 0.5 + rand() * 0.5;
        end
        
        % Adapt CR
        if ~isempty(CR_history)
            recent_CR = mean(CR_history(max(1, end-9):end));
            CR = max(0.1, min(0.9, recent_CR + randn() * 0.1));
        else
            CR = 0.5 + rand() * 0.4;
        end
        
        % Select strategy based on performance
        strategy_scores = mean(strategy_performance, 2);
        probabilities = strategy_scores / sum(strategy_scores);
        
        % Roulette wheel selection
        r = rand();
        cum_prob = 0;
        for i = 1:length(probabilities)
            cum_prob = cum_prob + probabilities(i);
            if r <= cum_prob
                strategy_idx = i;
                break;
            end
        end
    end
end

function mutant = apply_strategy(strategy_idx, pop, i, F, global_best_idx, fitness)
    % Apply selected mutation strategy
    pop_size = size(pop, 1);
    
    switch strategy_idx
        case 1 % DE/rand/1
            idxs = randperm(pop_size, 3);
            while any(idxs == i)
                idxs = randperm(pop_size, 3);
            end
            a = pop(idxs(1), :);
            b = pop(idxs(2), :);
            c = pop(idxs(3), :);
            mutant = a + F * (b - c);
            
        case 2 % DE/best/1
            [~, best_idx] = min(fitness);
            idxs = randperm(pop_size, 2);
            while any(idxs == i) || any(idxs == best_idx)
                idxs = randperm(pop_size, 2);
            end
            a = pop(best_idx, :);
            b = pop(idxs(1), :);
            c = pop(idxs(2), :);
            mutant = a + F * (b - c);
            
        case 3 % DE/rand/2
            idxs = randperm(pop_size, 5);
            while any(idxs == i)
                idxs = randperm(pop_size, 5);
            end
            a = pop(idxs(1), :);
            b = pop(idxs(2), :);
            c = pop(idxs(3), :);
            d = pop(idxs(4), :);
            e = pop(idxs(5), :);
            mutant = a + F * (b - c) + F * (d - e);
            
        case 4 % DE/current-to-best/1
            idxs = randperm(pop_size, 2);
            while any(idxs == i) || any(idxs == global_best_idx)
                idxs = randperm(pop_size, 2);
            end
            current = pop(i, :);
            best = pop(global_best_idx, :);
            r1 = pop(idxs(1), :);
            r2 = pop(idxs(2), :);
            mutant = current + F * (best - current) + F * (r1 - r2);
    end
end

%% Rastrigin 7D Function
function z = rastrigin_7d(x)
    % 7D Rastrigin function
    % Minimum at x = [0,0,...,0] with f(x) = 0
    n = length(x);
    z = 10 * n;
    for i = 1:n
        z = z + (x(i)^2 - 10 * cos(2 * pi * x(i)));
    end
end