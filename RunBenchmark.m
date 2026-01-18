close all; clear; clc;

%% 1. THIẾT LẬP HỆ THỐNG
fprintf('--- KHOI TAO HE THONG ---\n');
M = 12; 

% Tạo môi trường góc quét
Q = 160; phi = 1;
eqDir = -1:phi/Q:1-phi/Q;
Aq = generateQuantizedArrResponse(M, eqDir);

% Tạo Mẫu mong muốn (Target Pattern - PdM)
desDirs_c = 0.0;
[PdM, P_refGen, W0] = generateDesPattern(eqDir, sin(desDirs_c), Aq);
alpha = sort([find(ismember(eqDir, eqDir(1:4:end))), find(PdM)]);

%% 2. TẠO ULA 12 THÔNG THƯỜNG (CONVENTIONAL ULA)
% Mảng thông thường chỉ là vector trọng số toàn số 1 (chưa tối ưu)
W_ULA = ones(M, 1);

%% 3. CHẠY ILS (BASELINE)
fprintf('--- Dang chay ILS (Original Paper)... ---\n');
tic;
P_init = ones(size(eqDir));
% Gọi hàm ILS có sẵn
W_ILS = twoStepILS(50, alpha, Aq, W0, P_init, PdM); 
time_ILS = toc;
fprintf('ILS Hoan thanh: %.4fs\n', time_ILS);

%% 4. CHẠY CÁC BIẾN THỂ DE
strategies = {'rand1', 'rand2', 'best1', 'current_to_best1'};
results = struct();
colors = {'b', 'g', 'r', 'k'};       
lineStyles = {'--', '-.', ':', '-'}; 

fprintf('\n--- Dang chay so sanh 4 bien the DE... ---\n');
for i = 1:length(strategies)
    strat = strategies{i};
    fprintf('  [%d/4] Strategy: %s ... ', i, strat);
    tic;
    % Gọi hàm DESolver
    [w_best, best_cost, ~] = DESolver(Aq, PdM, alpha, M, strat);
    time_run = toc;
    
    % Lưu kết quả
    results(i).name = strat;
    results(i).w = w_best;
    results(i).time = time_run;
    results(i).cost = best_cost;
    fprintf('Done (%.2fs) | Cost: %.4f\n', time_run, best_cost);
end

%% 5. VẼ HÌNH SO SÁNH (BEAM PATTERN BENCHMARK)
figure('Name', 'Beam Pattern Benchmark', 'Position', [100, 100, 1000, 600]);
hold on; grid on;

% 1. Vẽ Mẫu mong muốn (Target) - Màu Tím
plot(eqDir, 10*log10(PdM/max(PdM)), 'm-*', 'LineWidth', 0.5, ...
    'MarkerIndices', 1:20:length(eqDir), 'DisplayName', 'Desired Pattern');

% 2. Vẽ ULA 12 Thông thường (Chưa tối ưu) - Nét đứt màu đen nhạt
pat_ULA = abs(W_ULA' * Aq);
plot(eqDir, 10*log10(pat_ULA/max(pat_ULA)), 'k--', ...
    'LineWidth', 1.0, 'DisplayName', 'Conventional ULA 12');

% 3. Vẽ ILS (Baseline) - Màu Xám đậm, nét liền đậm
pat_ILS = abs(W_ILS' * Aq);
plot(eqDir, 10*log10(pat_ILS/max(pat_ILS)), 'Color', [0.5 0.5 0.5], ...
    'LineWidth', 3, 'DisplayName', 'ILS (Original)');

% 4. Vẽ Các biến thể DE
for i = 1:length(strategies)
    w = results(i).w;
    pat = abs(w' * Aq);
    pat_norm = pat / max(pat);
    
    plot(eqDir, 10*log10(pat_norm), 'Color', colors{i}, ...
        'LineStyle', lineStyles{i}, 'LineWidth', 1.5, ...
        'DisplayName', ['DE/' results(i).name]);
end

xlabel("Equivalent directions"); ylabel("|A|, dB");
xlim([-1 1]); ylim([-50, 0]);
legend('Location', 'bestoutside', 'NumColumns', 2);
title('So sánh Búp sóng: ULA vs ILS vs DE Variants');

%% 6. XUẤT BẢNG KẾT QUẢ
fprintf('\n==================================================\n');
fprintf('%-20s | %-10s | %-10s\n', 'Algorithm', 'Time(s)', 'Final Cost');
fprintf('--------------------------------------------------\n');

% Tính Cost ILS
pat_ILS_norm = abs(W_ILS' * Aq); 
pat_ILS_norm = pat_ILS_norm/max(pat_ILS_norm);
diff_ILS = pat_ILS_norm(alpha) - PdM(alpha);
cost_ILS = sum(abs(diff_ILS).^2);

fprintf('%-20s | %-10s | %-10s\n', 'Conventional ULA', '-', '-'); % ULA không tính cost vì nó là tham chiếu
fprintf('%-20s | %-10.4f | %-10.4f\n', 'ILS (Original)', time_ILS, cost_ILS);
for i = 1:length(strategies)
    fprintf('%-20s | %-10.4f | %-10.4f\n', ['DE/' results(i).name], ...
        results(i).time, results(i).cost);
end
fprintf('==================================================\n');