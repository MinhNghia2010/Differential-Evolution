clc; clear; close all;

%% DE Parameters
pop_size = 30;
num_gen = 100;
dim = 2;         % Problem dimension (x, y)
lb = -5.12;
ub =  5.12;

% Tạo cell array để lưu kết quả của 5 biến thể (bao gồm adaptive)
mutation_strategies = {'DE/rand/1 (Original)', 'DE/best/1', 'DE/rand/2', 'DE/current-to-best/1', 'Adaptive DE'};
best_curves = cell(5, 1);
best_solutions = cell(5, 1);
best_fitnesses = zeros(5, 1);

%% Chạy cả 5 biến thể DE
for strategy = 1:5
    fprintf('Running %s...\n', mutation_strategies{strategy});
    
    %% Initialize Population
    pop = lb + (ub - lb) * rand(pop_size, dim);
    fitness = arrayfun(@(i) rastrigin(pop(i,1), pop(i,2)), 1:pop_size)';

    [best_fitness, idx] = min(fitness);
    best_solution = pop(idx, :);
    best_curve = zeros(num_gen, 1);
    
    % Khởi tạo biến cho adaptive DE
    if strategy == 5
        success_history = [];
        F_history = [];
        CR_history = [];
        strategy_performance = ones(4, 10); % Hiệu suất 4 chiến lược
        current_strategy = 1;
    end

    %% DE Main Loop
    for gen = 1:num_gen
        [~, global_best_idx] = min(fitness);
        successful_mutations = 0;
        
        for i = 1:pop_size
            %% ADAPTIVE DE: Tự động chọn tham số và chiến lược
            if strategy == 5
                [F, CR, current_strategy] = adaptive_parameters(...
                    gen, success_history, F_history, CR_history, strategy_performance);
            else
                % Fixed parameters for non-adaptive strategies
                F = 0.8;
                CR = 0.9;
            end
            
            % Mutation: Chọn chiến lược đột biến
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
                    
                case 5 % Adaptive DE - sử dụng chiến lược được chọn tự động
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
            trial_fit = rastrigin(trial(1), trial(2));
            if trial_fit < fitness(i)
                pop(i, :) = trial;
                fitness(i) = trial_fit;
                successful_mutations = successful_mutations + 1;
                
                % Cập nhật thông tin cho adaptive DE
                if strategy == 5
                    F_history = [F_history, F];
                    CR_history = [CR_history, CR];
                    % Cập nhật hiệu suất chiến lược
                    strategy_performance(current_strategy, mod(gen-1, 10)+1) = ...
                        strategy_performance(current_strategy, mod(gen-1, 10)+1) + 1;
                end
            end
        end

        % Cập nhật lịch sử thành công cho adaptive DE
        if strategy == 5
            success_rate = successful_mutations / pop_size;
            success_history = [success_history, success_rate];
            
            % Giữ kích thước lịch sử hợp lý
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
        
        % Hiển thị tiến trình cho adaptive DE
        if strategy == 5 && mod(gen, 20) == 0
            fprintf('  Generation %d: Best Fitness = %.4f, Success Rate = %.2f\n', ...
                    gen, best_fitness, success_rate);
        end
    end

    % Lưu kết quả
    best_curves{strategy} = best_curve;
    best_solutions{strategy} = best_solution;
    best_fitnesses(strategy) = best_fitness;
    
    % Hiển thị kết quả từng biến thể
    fprintf('%s - Best Solution: [%.4f, %.4f], Fitness: %.4f\n\n', ...
            mutation_strategies{strategy}, best_solution(1), best_solution(2), best_fitness);
end

%% Vẽ cả 5 đồ thị trên cùng 1 figure
figure;
colors = ['r', 'b', 'g', 'm', 'y'];
line_styles = {'-', '--', ':', '-.', '-'};
line_widths = [1.5, 1.5, 1.5, 1.5, 2.5]; % Adaptive DE nổi bật hơn

hold on;
for i = 1:5
    plot(1:num_gen, best_curves{i}, 'Color', colors(i), 'LineStyle', line_styles{i}, ...
         'LineWidth', line_widths(i), 'DisplayName', mutation_strategies{i});
end
hold off;

xlabel('Generation');
ylabel('Best Fitness');
title('DE Convergence Comparison: Original vs Variants vs Adaptive DE');
legend('show', 'Location', 'northeast');
grid on;

% Export all fitness data to CSV
all_fitness_data = [best_curves{1}, best_curves{2}, best_curves{3}, best_curves{4}, best_curves{5}];
csvwrite('DE_all_strategies_with_adaptive_fitness.csv', all_fitness_data);

%% Hiển thị kết quả tổng hợp
fprintf('\n=== COMPARISON RESULTS ===\n');
for i = 1:5
    fprintf('%s: %.6f\n', mutation_strategies{i}, best_fitnesses(i));
end

[best_overall_fitness, best_idx] = min(best_fitnesses);
fprintf('\nBEST PERFORMER: %s with fitness = %.6f\n', ...
        mutation_strategies{best_idx}, best_overall_fitness);

%% Adaptive DE Functions
function [F, CR, strategy_idx] = adaptive_parameters(gen, success_history, F_history, CR_history, strategy_performance)
    % Adaptive parameter and strategy selection
    
    if isempty(success_history)
        % Khởi tạo ngẫu nhiên
        F = 0.5 + rand() * 0.5;
        CR = 0.5 + rand() * 0.4;
        strategy_idx = randi(4);
    else
        % Thích ứng F dựa trên lịch sử thành công
        if ~isempty(F_history)
            recent_success = mean(success_history(max(1, end-4):end));
            if recent_success > 0.1
                % Nếu thành công, thử F mạnh hơn
                F = 0.6 + rand() * 0.4;
            else
                % Nếu ít thành công, thử F khác
                F = 0.3 + rand() * 0.4;
            end
        else
            F = 0.5 + rand() * 0.5;
        end
        
        % Thích ứng CR
        if ~isempty(CR_history)
            recent_CR = mean(CR_history(max(1, end-9):end));
            CR = max(0.1, min(0.9, recent_CR + randn() * 0.1));
        else
            CR = 0.5 + rand() * 0.4;
        end
        
        % Chọn chiến lược dựa trên hiệu suất
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
    % Áp dụng chiến lược đột biến được chọn
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

%% Rastrigin Function
function z = rastrigin(x, y)
    z = 20 + x.^2 + y.^2 - 10 * (cos(2 * pi * x) + cos(2 * pi * y));
end
