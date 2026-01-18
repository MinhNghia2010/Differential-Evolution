% -------------------------------------------------------------------------
% Author: Nnaji Obinna
% Student ID: 101001461
% Institution: Ontario Tech University
% Project: Rastrigin Function Optimization using GA, DE, and PSO
% Description: DE variants comparison on 7D Rastrigin function (Statistical Version)
% Copyright (c) 2025 Nnaji Obinna & Ontario Tech University.
% All rights reserved.
% -------------------------------------------------------------------------
function DERastrigin7D_2D()
clc; clear; close all;

fprintf('=== RUNNING DE ON 7D RASTRIGIN FUNCTION (STATISTICAL MODE) ===\n');

%% DE Parameters for 7D
pop_size = 70;       % 10 * dim is standard
num_gen = 500;       % Increased slightly for 7D to allow better convergence
dim = 7;
lb = -5.12;
ub = 5.12;
num_runs = 30;       % STANDARD: 30 independent runs

% Strategies to compare
mutation_strategies = {'DE/rand/1 (Original)', 'DE/best/1', 'DE/rand/2', 'DE/current-to-best/1', 'Adaptive DE'};
num_strategies = length(mutation_strategies);

% Storage for results
avg_curves = cell(num_strategies, 1);
stats_results = zeros(num_strategies, 2); % Col 1: Mean, Col 2: Std
avg_times = zeros(num_strategies, 1);

%% Run all 5 DE variants
for strategy = 1:num_strategies
    fprintf('Running %s (%d runs)...\n', mutation_strategies{strategy}, num_runs);
    
    all_runs_curves = zeros(num_gen, num_runs);
    final_fitnesses = zeros(num_runs, 1);
    run_times = zeros(num_runs, 1);
    
    %% Loop for Independent Runs
    for run = 1:num_runs
        tic;
        
        % Initialize Population (Reset for each run)
        pop = lb + (ub - lb) * rand(pop_size, dim);
        fitness = zeros(pop_size, 1);
        for i = 1:pop_size
            fitness(i) = rastrigin_7d_2d(pop(i, :));
        end

        [current_best, idx] = min(fitness);
        % best_solution = pop(idx, :); % Uncomment if you need to store solutions
        best_curve = zeros(num_gen, 1);
        
        % Initialize variables for adaptive DE (Reset for each run)
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
                    F = 0.5; % 0.5 is often better for Rastrigin than 0.8
                    CR = 0.9;
                end
                
                % Mutation: Select strategy
                switch strategy
                    case 1 % DE/rand/1 (Original)
                        idxs = randperm(pop_size, 3);
                        while any(idxs == i), idxs = randperm(pop_size, 3); end
                        mutant = pop(idxs(1),:) + F * (pop(idxs(2),:) - pop(idxs(3),:));
                        
                    case 2 % DE/best/1
                        [~, best_idx] = min(fitness);
                        idxs = randperm(pop_size, 2);
                        while any(idxs == i) || any(idxs == best_idx), idxs = randperm(pop_size, 2); end
                        mutant = pop(best_idx,:) + F * (pop(idxs(1),:) - pop(idxs(2),:));
                        
                    case 3 % DE/rand/2
                        idxs = randperm(pop_size, 5);
                        while any(idxs == i), idxs = randperm(pop_size, 5); end
                        mutant = pop(idxs(1),:) + F * (pop(idxs(2),:) - pop(idxs(3),:)) ...
                                                + F * (pop(idxs(4),:) - pop(idxs(5),:));
                        
                    case 4 % DE/current-to-best/1
                        idxs = randperm(pop_size, 2);
                        while any(idxs == i) || any(idxs == global_best_idx), idxs = randperm(pop_size, 2); end
                        mutant = pop(i,:) + F * (pop(global_best_idx,:) - pop(i,:)) ...
                                          + F * (pop(idxs(1),:) - pop(idxs(2),:));
                        
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
                trial_fit = rastrigin_7d_2d(trial);
                
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
                
                if length(success_history) > 20, success_history = success_history(end-19:end); end
                if length(F_history) > 50, F_history = F_history(end-49:end); end
                if length(CR_history) > 50, CR_history = CR_history(end-49:end); end
            end

            % Track best solution
            current_iter_best = min(fitness);
            if current_iter_best < current_best
                current_best = current_iter_best;
            end
            best_curve(gen) = current_best;
        end
        
        % Store single run data
        run_times(run) = toc;
        final_fitnesses(run) = current_best;
        all_runs_curves(:, run) = best_curve;
    end
    
    % Calculate Statistics for this Strategy
    avg_curves{strategy} = mean(all_runs_curves, 2);
    stats_results(strategy, 1) = mean(final_fitnesses); % Mean
    stats_results(strategy, 2) = std(final_fitnesses);  % Std Dev
    avg_times(strategy) = mean(run_times);
    
    fprintf('   -> Finished. Mean: %.4e, Std: %.4e, Avg Time: %.2fs\n', ...
            stats_results(strategy, 1), stats_results(strategy, 2), avg_times(strategy));
end

%% Plot results
figure('Color', 'w');
colors = {'r', 'b', 'g', 'm', [0.9 0.6 0]}; % Using cell for colors to handle RGB triplet
line_styles = {'-', '--', ':', '-.', '-'};
markers = {'none', 'none', 'none', 'none', 'o'};
line_widths = [1.5, 1.5, 1.5, 1.5, 2];

hold on;
for i = 1:5
    semilogy(1:num_gen, avg_curves{i}, ...
        'Color', colors{i}, ...
        'LineStyle', line_styles{i}, ...
        'LineWidth', line_widths(i), ...
        'Marker', markers{i}, ...
        'MarkerIndices', 1:20:num_gen, ...
        'DisplayName', mutation_strategies{i});
end
hold off;

xlabel('Generation');
ylabel('Mean Best Fitness (Log Scale)');
title(['DE Convergence on 7D Rastrigin (Avg of ', num2str(num_runs), ' runs)']);
legend('show', 'Location', 'southwest');
grid on;
box on;

%% Display summary results
fprintf('\n==========================================================================\n');
fprintf('FINAL STATISTICAL RESULTS FOR 7D RASTRIGIN (%d RUNS)\n', num_runs);
fprintf('==========================================================================\n');
fprintf('%-25s | %-12s | %-12s | %-10s\n', 'Strategy', 'Mean Fit', 'Std Dev', 'Avg Time(s)');
fprintf('--------------------------------------------------------------------------\n');
for i = 1:5
    fprintf('%-25s | %.4e   | %.4e   | %-10.2f\n', ...
        mutation_strategies{i}, stats_results(i, 1), stats_results(i, 2), avg_times(i));
end
fprintf('--------------------------------------------------------------------------\n');

[best_mean, best_idx] = min(stats_results(:, 1));
fprintf('\nBEST PERFORMER (On Average): %s with Mean Fitness = %.6e\n', ...
        mutation_strategies{best_idx}, best_mean);

end

%% Adaptive DE Functions
function [F, CR, strategy_idx] = adaptive_parameters(gen, success_history, F_history, CR_history, strategy_performance)
    if isempty(success_history)
        F = 0.5 + rand() * 0.5;
        CR = 0.5 + rand() * 0.4;
        strategy_idx = randi(4);
    else
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
        
        if ~isempty(CR_history)
            recent_CR = mean(CR_history(max(1, end-9):end));
            CR = max(0.1, min(0.9, recent_CR + randn() * 0.1));
        else
            CR = 0.5 + rand() * 0.4;
        end
        
        strategy_scores = mean(strategy_performance, 2);
        probabilities = strategy_scores / sum(strategy_scores);
        
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
    pop_size = size(pop, 1);
    switch strategy_idx
        case 1 
            idxs = randperm(pop_size, 3);
            while any(idxs == i), idxs = randperm(pop_size, 3); end
            mutant = pop(idxs(1), :) + F * (pop(idxs(2), :) - pop(idxs(3), :));
        case 2 
            [~, best_idx] = min(fitness);
            idxs = randperm(pop_size, 2);
            while any(idxs == i) || any(idxs == best_idx), idxs = randperm(pop_size, 2); end
            mutant = pop(best_idx, :) + F * (pop(idxs(1), :) - pop(idxs(2), :));
        case 3
            idxs = randperm(pop_size, 5);
            while any(idxs == i), idxs = randperm(pop_size, 5); end
            mutant = pop(idxs(1), :) + F * (pop(idxs(2), :) - pop(idxs(3), :)) + F * (pop(idxs(4), :) - pop(idxs(5), :));
        case 4 
            idxs = randperm(pop_size, 2);
            while any(idxs == i) || any(idxs == global_best_idx), idxs = randperm(pop_size, 2); end
            mutant = pop(i, :) + F * (pop(global_best_idx, :) - pop(i, :)) + F * (pop(idxs(1), :) - pop(idxs(2), :));
    end
end

%% Rastrigin 7D Function
function z = rastrigin_7d_2d(x)
    % 7D Rastrigin function
    n = length(x);
    z = 10 * n;
    for i = 1:n
        z = z + (x(i)^2 - 10 * cos(2 * pi * x(i)));
    end
end