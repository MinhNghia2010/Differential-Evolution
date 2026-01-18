function [W_best, BestCost, CostHistory] = ADESolver(Aq, PdM, alpha, M)
    % JADE: Joint Adaptive Differential Evolution
    % Strategy: DE/current-to-pbest/1 with parameter adaptation
    
    % --- THAM SỐ CẤU HÌNH ---
    NP = 50;            % Kích thước quần thể
    MaxIter = 1000;     % Số vòng lặp
    c = 0.1;            % Tốc độ học của tham số (Learning rate)
    p = 0.05;           % Tỷ lệ top p-best (ví dụ: top 5%)
    
    % Khởi tạo tham số thích nghi
    mu_CR = 0.5;        % Trung bình khởi tạo của CR
    mu_F = 0.5;         % Trung bình khởi tạo của F
    
    % --- KHỞI TẠO QUẦN THỂ ---
    Dim = 2 * M;
    LowerBound = -1; UpperBound = 1;
    Pop = LowerBound + (UpperBound - LowerBound) .* rand(NP, Dim);
    
    Cost = zeros(NP, 1);
    for i = 1:NP
        Cost(i) = CostFunction(Pop(i, :), Aq, PdM, alpha, M);
    end
    
    [BestCost, BestIdx] = min(Cost);
    BestSol = Pop(BestIdx, :);
    CostHistory = zeros(MaxIter, 1);
    
    % --- VÒNG LẶP TIẾN HÓA ---
    for it = 1:MaxIter
        S_CR = []; % Lưu các giá trị CR thành công trong thế hệ này
        S_F = [];  % Lưu các giá trị F thành công trong thế hệ này
        
        % Sắp xếp quần thể để tìm p-best
        [~, sorted_idx] = sort(Cost);
        
        for i = 1:NP
            % 1. SINH THAM SỐ F và CR CHO CÁ THỂ i
            % CR theo phân phối Chuẩn (Normal): N(mu_CR, 0.1)
            CRi = mu_CR + 0.1 * randn;
            CRi = min(1, max(0, CRi)); % Kẹp trong [0, 1]
            
            % F theo phân phối Cauchy: C(mu_F, 0.1)
            % Cách tạo Cauchy: mu + gamma * tan(pi * (rand - 0.5))
            Fi = mu_F + 0.1 * tan(pi * (rand - 0.5));
            while Fi <= 0
                Fi = mu_F + 0.1 * tan(pi * (rand - 0.5)); % Sinh lại nếu <= 0
            end
            Fi = min(1, Fi); % Kẹp trần tại 1
            
            % 2. MUTATION: current-to-pbest/1
            % V = Xi + F*(X_pbest - Xi) + F*(Xr1 - Xr2)
            
            % Chọn X_pbest ngẫu nhiên từ top p% (top 100*p cá thể tốt nhất)
            num_pbest = max(1, round(p * NP));
            pbest_idx = sorted_idx(randi(num_pbest));
            x_pbest = Pop(pbest_idx, :);
            
            % Chọn r1, r2 khác i
            idxs = randperm(NP, 3);
            idxs(idxs == i) = [];
            r1 = idxs(1); r2 = idxs(2);
            
            x_i = Pop(i, :);
            x_r1 = Pop(r1, :);
            x_r2 = Pop(r2, :);
            
            Mutant = x_i + Fi * (x_pbest - x_i) + Fi * (x_r1 - x_r2);
            
            % 3. CROSSOVER
            Cross_Mask = rand(1, Dim) <= CRi;
            j_rand = randi(Dim);
            Cross_Mask(j_rand) = true;
            
            Trial = x_i;
            Trial(Cross_Mask) = Mutant(Cross_Mask);
            
            % Kiểm tra biên
            Trial = max(Trial, LowerBound);
            Trial = min(Trial, UpperBound);
            
            % 4. SELECTION
            TrialCost = CostFunction(Trial, Aq, PdM, alpha, M);
            
            if TrialCost < Cost(i)
                Pop(i, :) = Trial;
                Cost(i) = TrialCost;
                
                % Lưu tham số thành công
                S_CR = [S_CR; CRi];
                S_F = [S_F; Fi];
                
                if TrialCost < BestCost
                    BestCost = TrialCost;
                    BestSol = Trial;
                end
            end
        end
        
        % 5. CẬP NHẬT THAM SỐ THÍCH NGHI (mu_CR, mu_F)
        if ~isempty(S_CR)
            % Cập nhật mu_CR (trung bình cộng)
            mu_CR = (1 - c) * mu_CR + c * mean(S_CR);
            
            % Cập nhật mu_F (trung bình Lehmer)
            mean_Lehmer = sum(S_F.^2) / sum(S_F);
            mu_F = (1 - c) * mu_F + c * mean_Lehmer;
        end
        
        CostHistory(it) = BestCost;
    end
    
    % Chuyển kết quả
    w_real = BestSol(1:M);
    w_imag = BestSol(M+1:end);
    W_best = (w_real + 1i * w_imag).';
end

function cost = CostFunction(x, Aq, PdM, alpha, M)
    w_real = x(1:M);
    w_imag = x(M+1:end);
    w = (w_real + 1i * w_imag).';
    
    GeneratedPattern = abs(w' * Aq);
    max_val = max(GeneratedPattern);
    if max_val > 0
         GeneratedPattern = GeneratedPattern / max_val;
    end
    
    diff = GeneratedPattern(alpha) - PdM(alpha);
    cost = sum(abs(diff).^2); 
end