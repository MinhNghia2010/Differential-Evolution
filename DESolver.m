function [W_best, BestCost, CostHistory] = DESolver(Aq, PdM, alpha, M, strategy)
    % strategy options: 'rand1', 'rand2', 'best1', 'current_to_best1'
    
    % --- THAM SỐ DE ---
    NP = 50;            % Kích thước quần thể
    MaxIter = 1000;     % Số vòng lặp
    F = 0.5;            % Trọng số lai ghép
    CR = 0.9;           % Tỷ lệ trao đổi chéo
    
    % --- KHỞI TẠO ---
    Dim = 2 * M;        % Biến tối ưu (Thực + Ảo)
    LowerBound = -1;    
    UpperBound = 1;     
    
    % Quần thể ngẫu nhiên
    Pop = LowerBound + (UpperBound - LowerBound) .* rand(NP, Dim);
    Cost = zeros(NP, 1);
    
    % Đánh giá ban đầu
    for i = 1:NP
        Cost(i) = CostFunction(Pop(i, :), Aq, PdM, alpha, M);
    end
    
    % Tìm Global Best ban đầu
    [BestCost, BestIdx] = min(Cost);
    BestSol = Pop(BestIdx, :);
    CostHistory = zeros(MaxIter, 1); 
    
    % --- VÒNG LẶP TIẾN HÓA ---
    for it = 1:MaxIter
        for i = 1:NP
            % 1. ĐỘT BIẾN (MUTATION)
            % Chọn r1...r5 khác i.
            % SỬA LỖI: Lấy 6 số để trừ hao nếu trùng i thì vẫn còn đủ 5 số
            idxs = randperm(NP, 6); 
            idxs(idxs == i) = []; 
            
            % Gán các chỉ số (đảm bảo không bị lỗi index exceed)
            r1 = idxs(1); r2 = idxs(2); r3 = idxs(3); r4 = idxs(4); r5 = idxs(5);
            
            x_i = Pop(i, :);
            x_r1 = Pop(r1, :); x_r2 = Pop(r2, :); x_r3 = Pop(r3, :);
            
            switch strategy
                case 'rand1'
                    % V = Xr1 + F(Xr2 - Xr3)
                    Mutant = x_r1 + F * (x_r2 - x_r3);
                    
                case 'rand2'
                    % V = Xr1 + F(Xr2 - Xr3) + F(Xr4 - Xr5)
                    x_r4 = Pop(r4, :); x_r5 = Pop(r5, :);
                    Mutant = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5);
                    
                case 'best1'
                    % V = Best + F(Xr1 - Xr2)
                    Mutant = BestSol + F * (x_r1 - x_r2);
                    
                case 'current_to_best1'
                    % V = Xi + F(Best - Xi) + F(Xr1 - Xr2)
                    Mutant = x_i + F * (BestSol - x_i) + F * (x_r1 - x_r2);
            end
            
            % 2. LAI GHÉP (CROSSOVER)
            Cross_Mask = rand(1, Dim) <= CR;
            j_rand = randi(Dim); 
            Cross_Mask(j_rand) = true;
            
            Trial = x_i;
            Trial(Cross_Mask) = Mutant(Cross_Mask);
            
            % 3. KIỂM TRA BIÊN (CLAMPING)
            Trial = max(Trial, LowerBound);
            Trial = min(Trial, UpperBound);
            
            % 4. CHỌN LỌC (SELECTION)
            TrialCost = CostFunction(Trial, Aq, PdM, alpha, M);
            
            if TrialCost < Cost(i)
                Pop(i, :) = Trial;
                Cost(i) = TrialCost;
                
                % Cập nhật Global Best
                if TrialCost < BestCost
                    BestCost = TrialCost;
                    BestSol = Trial;
                end
            end
        end
        CostHistory(it) = BestCost;
    end
    
    % --- KẾT QUẢ ---
    w_real = BestSol(1:M);
    w_imag = BestSol(M+1:end);
    W_best = (w_real + 1i * w_imag).'; 
end

% --- HÀM MỤC TIÊU ---
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