close all; clear; clc;

%% 1. THIẾT LẬP HỆ THỐNG
fprintf('--- KHOI TAO HE THONG ---\n');
M = 12; 
Q = 160; phi = 1;
eqDir = -1:phi/Q:1-phi/Q;
Aq = generateQuantizedArrResponse(M, eqDir);

% Mẫu mong muốn
desDirs_c = 0.0;
[PdM, P_refGen, W0] = generateDesPattern(eqDir, sin(desDirs_c), Aq);
alpha = sort([find(ismember(eqDir, eqDir(1:4:end))), find(PdM)]);

% ULA thường
W_ULA = ones(M, 1);

%% 2. CHẠY ILS (BASELINE)
fprintf('1. Dang chay ILS (Original Paper)...\n');
tic;
P_init = ones(size(eqDir));
W_ILS = twoStepILS(50, alpha, Aq, W0, P_init, PdM); 
time_ILS = toc;
fprintf('   -> Xong: %.4fs\n', time_ILS);

%% 3. CHẠY DE TRUYỀN THỐNG (DE/rand/1)
fprintf('2. Dang chay DE Standard (DE/rand/1)...\n');
tic;
% Gọi hàm DESolver cũ (bạn cần đảm bảo file DESolver.m vẫn còn trong thư mục)
[W_DE, cost_DE, ~] = DESolver(Aq, PdM, alpha, M, 'rand1');
time_DE = toc;
fprintf('   -> Xong: %.4fs | Cost: %.4f\n', time_DE, cost_DE);

%% 4. CHẠY ADAPTIVE DE (JADE)
fprintf('3. Dang chay Adaptive DE (JADE)...\n');
tic;
% Gọi hàm ADESolver mới
[W_ADE, cost_ADE, ~] = ADESolver(Aq, PdM, alpha, M);
time_ADE = toc;
fprintf('   -> Xong: %.4fs | Cost: %.4f\n', time_ADE, cost_ADE);

%% 5. VẼ HÌNH SO SÁNH
figure('Name', 'Adaptive DE vs ILS Benchmark', 'Position', [100, 100, 1000, 600]);
hold on; grid on;

% 1. Target (Tím)
plot(eqDir, 10*log10(PdM/max(PdM)), 'm-*', 'LineWidth', 0.5, ...
    'MarkerIndices', 1:20:length(eqDir), 'DisplayName', 'Desired Pattern');

% 2. ULA (Đen nét đứt)
pat_ULA = abs(W_ULA' * Aq);
plot(eqDir, 10*log10(pat_ULA/max(pat_ULA)), 'k--', 'LineWidth', 1, ...
    'DisplayName', 'Conventional ULA');

% 3. ILS (Xám đậm)
pat_ILS = abs(W_ILS' * Aq);
plot(eqDir, 10*log10(pat_ILS/max(pat_ILS)), 'Color', [0.4 0.4 0.4], ...
    'LineWidth', 3, 'DisplayName', 'ILS (Baseline)');

% 4. DE Standard (Xanh dương)
pat_DE = abs(W_DE' * Aq);
plot(eqDir, 10*log10(pat_DE/max(pat_DE)), 'b-.', 'LineWidth', 1.5, ...
    'DisplayName', 'Standard DE (rand/1)');

% 5. Adaptive DE (Đỏ) - ĐỐI TƯỢNG CHÍNH
pat_ADE = abs(W_ADE' * Aq);
plot(eqDir, 10*log10(pat_ADE/max(pat_ADE)), 'r', 'LineWidth', 1.8, ...
    'DisplayName', 'Adaptive DE (JADE)');

xlabel("Equivalent directions"); ylabel("|A|, dB");
xlim([-1 1]); ylim([-60, 0]); % Zoom sâu hơn để thấy sự ưu việt của ADE
legend('Location', 'bestoutside');
title('So sánh hiệu năng: ILS vs DE vs Adaptive DE (JADE)');

%% 6. BẢNG KẾT QUẢ
fprintf('\n==================================================\n');
fprintf('%-20s | %-10s | %-10s\n', 'Algorithm', 'Time(s)', 'Final Cost');
fprintf('--------------------------------------------------\n');

% Tính cost ILS
pat_ILS_norm = abs(W_ILS' * Aq); pat_ILS_norm = pat_ILS_norm/max(pat_ILS_norm);
diff_ILS = pat_ILS_norm(alpha) - PdM(alpha);
cost_ILS = sum(abs(diff_ILS).^2);

fprintf('%-20s | %-10s | %-10s\n', 'Conventional ULA', '-', '-');
fprintf('%-20s | %-10.4f | %-10.4f\n', 'ILS (Original)', time_ILS, cost_ILS);
fprintf('%-20s | %-10.4f | %-10.4f\n', 'DE (rand/1)', time_DE, cost_DE);
fprintf('%-20s | %-10.4f | %-10.4f\n', 'Adaptive DE (JADE)', time_ADE, cost_ADE);
fprintf('==================================================\n');