close all;
clear;
clc;

%% 1. THIẾT LẬP HỆ THỐNG
fprintf('--- KHỞI TẠO HỆ THỐNG ---\n');
theta = (-90:0.1:90-0.1)*pi/180; % Góc thực tế
lambda = 1; 
M = 12; % Số phần tử ăng-ten

% Vector lái (Steering Vector) cho góc thực
A = generateSteeringVector(theta, M, lambda);

% Hướng tương đương (Equivalent Directions) cho quá trình tối ưu
Q = 160; 
phi = 1;
eqDir = -1:phi/Q:1-phi/Q;
Aq = generateQuantizedArrResponse(M, eqDir);

% Tạo Mẫu mong muốn (Target Pattern)
desDirs_c = 0.0;
[PdM, P_refGen, W0] = generateDesPattern(eqDir, sin(desDirs_c), Aq);

% Tạo vector trọng số cho ULA thông thường (để vẽ Hình 1)
W_ULA_Ref = ones(M, 1); 

% Xác định các điểm quan trọng cần tối ưu (alpha)
alpha = sort([find(ismember(eqDir, eqDir(1:4:end))), find(PdM)]);

%% 2. CHẠY TỐI ƯU HÓA: ILS vs DE

% --- A. Phương pháp ILS ---
fprintf('--- Đang chạy ILS (Iterative Least Squares)... ---\n');
tic;
P_init = ones(size(eqDir));
PM = P_init;
W_ref_ILS = twoStepILS(50, alpha, Aq, W0, PM, PdM); 
time_ILS = toc;
fprintf('ILS hoàn thành. Thời gian: %.4f s\n\n', time_ILS);

% --- B. Phương pháp DE ---
fprintf('--- Đang chạy DE (Differential Evolution)... ---\n');
tic;
% Gọi hàm DE (Đảm bảo tên hàm đúng với file bạn lưu)
[W_ref_DE, Cost_DE] = DE_Optimize_BF_compare(Aq, PdM, alpha, M);
time_DE = toc;
fprintf('DE hoàn thành. Thời gian: %.4f s\n', time_DE);


%% 3. VẼ HÌNH 1: SO SÁNH BÚP SÓNG THAM CHIẾU (FULL COMPARISON)
figure('Name', 'Fig 1: Reference Beam Comparison', 'Position', [100, 100, 900, 500]);
hold on

% 1. Vẽ ULA thông thường (Nền chuẩn)
pat_ULA = abs(W_ULA_Ref' * Aq);
plot(eqDir, 10*log10(pat_ULA/max(pat_ULA)), '--k', 'LineWidth', 1.0, 'DisplayName', 'Conventional ULA');

% 2. Vẽ Mẫu mong muốn (Target)
plot(eqDir, 10*log10(PdM/max(PdM)), 'm-*', 'MarkerIndices', 1:15:length(eqDir), 'DisplayName', 'Desired Pattern');

% 3. Vẽ kết quả ILS
pat_ILS = abs(W_ref_ILS'*Aq);
plot(eqDir, 10*log10(pat_ILS/max(pat_ILS)), 'b', 'LineWidth', 1.5, 'DisplayName', 'ILS Optimized');

% 4. Vẽ kết quả DE
pat_DE = abs(W_ref_DE'*Aq);
plot(eqDir, 10*log10(pat_DE/max(pat_DE)), 'r-.', 'LineWidth', 1.5, 'DisplayName', 'DE Optimized');

legend('Location', 'southoutside', 'NumColumns', 4);
xlabel("Equivalent directions"); ylabel("|A|, dB");
xlim([-1 1]); ylim([-40, 0]);
grid on;
title('Hình 1: So sánh Búp sóng Tham chiếu (ULA vs ILS vs DE)');


%% 4. TẠO CÁC BÚP SÓNG QUÉT (SCANNING BEAMS)
spacing = 0.2;
deltas = -0.8:spacing:0.8; 

W_dd_ILS = zeros(M, size(deltas, 2));
W_dd_DE  = zeros(M, size(deltas, 2));

for i=1:size(deltas, 2)
    W_dd_ILS(:, i) = displacePattern(W_ref_ILS, deltas(i), M);
    W_dd_DE(:, i)  = displacePattern(W_ref_DE, deltas(i), M);
end


%% 5. VẼ HÌNH 2: SO SÁNH QUÉT (ILS vs DE)
% CẬP NHẬT: Thay ULA bằng ILS để so sánh độ khớp giữa 2 thuật toán
figure('Name', 'Fig 2: Scanning Beams (ILS vs DE)', 'Position', [150, 150, 900, 500]);
hold on

% Vẽ các búp sóng ILS (Màu xanh dương, nét liền)
for i = 1:size(deltas, 2)
    pat = abs(W_dd_ILS(:, i)'*Aq);
    h1 = plot(eqDir, 10*log10(pat/max(pat)), 'b', 'LineWidth', 1.0); 
    if i~=1, set(h1, 'HandleVisibility', 'off'); end % Chỉ hiện legend 1 lần
end

% Vẽ các búp sóng DE (Màu đỏ, nét đứt hoặc chấm gạch để dễ nhìn khi chồng lên nhau)
for i = 1:size(deltas, 2)
    pat = abs(W_dd_DE(:, i)'*Aq);
    h2 = plot(eqDir, 10*log10(pat/max(pat)), 'r-.', 'LineWidth', 1.5); 
    if i~=1, set(h2, 'HandleVisibility', 'off'); end
end

legend('ILS Optimized (Scanning)', 'DE Optimized (Scanning)', 'Location', 'southoutside');
xlabel("Equivalent directions"); ylabel("|A|, dB");
xlim([-1 1]); ylim([-35, 0]);
grid on;
title('Hình 2: So sánh Búp sóng Quét (ILS vs DE)');


%% 6. KẾT HỢP ĐA BÚP SÓNG (MULTIBEAM COMBINATION)
ro = 0.5; 
comBeamIdx = cast(size(deltas, 2)/2, 'uint32');
senseBeamIdx = 2; 

% Chuẩn hóa
W_dd_ILS = W_dd_ILS./vecnorm(W_dd_ILS, 2, 2);
W_dd_DE  = W_dd_DE./vecnorm(W_dd_DE, 2, 2);

% Tính tổng hợp
W_final_ILS = sqrt(ro)*W_dd_ILS(:, comBeamIdx) + sqrt(1-ro)*W_dd_ILS(:, senseBeamIdx);
W_final_DE  = sqrt(ro)*W_dd_DE(:, comBeamIdx)  + sqrt(1-ro)*W_dd_DE(:, senseBeamIdx);

%% 7. VẼ HÌNH 3: SO SÁNH TỔNG HỢP (ILS vs DE)
figure('Name', 'Fig 3: Combined Multibeam Comparison', 'Position', [200, 200, 900, 500]);
hold on;

resp_final_ILS = abs(W_final_ILS' * A);
resp_final_DE  = abs(W_final_DE' * A);

plot(theta*180/pi, 10*log10(resp_final_ILS/max(resp_final_ILS)), 'b', 'LineWidth', 1.5, 'DisplayName', 'ILS Combined');
plot(theta*180/pi, 10*log10(resp_final_DE/max(resp_final_DE)), 'r-.', 'LineWidth', 1.5, 'DisplayName', 'DE Combined');

xlabel("\theta (Degree)"); ylabel("|A(\theta)|, dB");
grid on;
xlim([-90, 90]); ylim([-25, 0]);
legend('Location', 'southoutside');
title('Hình 3: So sánh Đa búp sóng Tổng hợp (ILS vs DE)');
subtitle(['Communication at 0^\circ, Sensing at approx ' num2str(asin(deltas(senseBeamIdx))*180/pi, '%.1f') '^\circ']);

fprintf('\n--- HOÀN TẤT VẼ HÌNH ---\n');

%% 7. VẼ HÌNH 3: TẬP HỢP ĐA BÚP SÓNG (SET OF MULTIBEAMS)
% Thay vì vẽ 1 đường, ta vẽ vòng lặp cho tất cả các hướng quét
figure('Name', 'Fig 3: Set of Multibeams Comparison', 'Position', [200, 100, 1000, 700]);

% --- SUBPLOT 1: KẾT QUẢ CỦA ILS ---
subplot(2, 1, 1);
hold on;
title('a) Tập đa búp sóng của phương pháp ILS (Tham chiếu)');
grid on;
xlim([-90, 90]); ylim([-25, 2]); % Mở rộng Y một chút để nhìn rõ đỉnh
xlabel("\theta (Degree)"); ylabel("|A(\theta)|, dB");

% Vòng lặp quét qua tất cả các hướng
for i = 1:size(W_dd_ILS, 2)
    % Bỏ qua trường hợp trùng với búp sóng liên lạc (thường là ở giữa)
    if i == comBeamIdx
        continue; 
    end
    
    % Trộn sóng: Com (Cố định) + Sense (Thay đổi theo i)
    w_combined = sqrt(ro)*W_dd_ILS(:, comBeamIdx) + sqrt(1-ro)*W_dd_ILS(:, i);
    
    % Tính và vẽ
    resp = abs(w_combined' * A);
    % Chuẩn hóa theo đỉnh cao nhất của búp sóng LIÊN LẠC
    % (Lưu ý: Búp sóng liên lạc sẽ cao hơn búp sóng cảm biến)
    resp_norm = resp / max(resp); 
    
    plot(theta*180/pi, 10*log10(resp_norm), 'b', 'LineWidth', 1.0);
end
% Vẽ một đường giả để hiện legend đại diện
plot(nan, nan, 'b', 'LineWidth', 1.5, 'DisplayName', 'ILS Beams');
legend('ILS Scans');


% --- SUBPLOT 2: KẾT QUẢ CỦA DE ---
subplot(2, 1, 2);
hold on;
title('b) Tập đa búp sóng của phương pháp DE (Đề xuất)');
grid on;
xlim([-90, 90]); ylim([-25, 2]);
xlabel("\theta (Degree)"); ylabel("|A(\theta)|, dB");

% Vòng lặp tương tự cho DE
for i = 1:size(W_dd_DE, 2)
    if i == comBeamIdx
        continue; 
    end
    
    w_combined = sqrt(ro)*W_dd_DE(:, comBeamIdx) + sqrt(1-ro)*W_dd_DE(:, i);
    
    resp = abs(w_combined' * A);
    resp_norm = resp / max(resp);
    
    plot(theta*180/pi, 10*log10(resp_norm), 'r', 'LineWidth', 1.0);
end
plot(nan, nan, 'r', 'LineWidth', 1.5, 'DisplayName', 'DE Beams');
legend('DE Scans');

fprintf('\n--- Đã vẽ xong Tập đa búp sóng ---\n');