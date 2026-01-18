function [W_best, BestCost] = DE_Optimize_BF_compare(Aq, PdM, alpha, M)
    % --- THAM SỐ DE ---
    NP = 50;            % Số lượng cá thể (Population Size)
    MaxIter = 1500;     % Số vòng lặp (Tăng lên nếu muốn kết quả tốt hơn)
    F = 0.5;            % Hệ số khuếch đại (Scaling Factor)
    CR = 0.9;           % Tỷ lệ lai ghép (Crossover Rate)
    
    % --- KHỞI TẠO ---
    % Biến tối ưu: Vector trọng số w (M số phức) -> Tách thành 2*M số thực
    Dim = 2 * M;        
    LowerBound = -1;    % Giới hạn không gian tìm kiếm
    UpperBound = 1;     
    
    % Quần thể ngẫu nhiên
    Pop = LowerBound + (UpperBound - LowerBound) .* rand(NP, Dim);
    Cost = zeros(NP, 1);
    
    % Tính chi phí ban đầu
    for i = 1:NP
        Cost(i) = CostFunction(Pop(i, :), Aq, PdM, alpha, M);
    end
    
    % Tìm cá thể tốt nhất (Global Best)
    [BestCost, BestIdx] = min(Cost);
    BestSol = Pop(BestIdx, :);
    
    % --- VÒNG LẶP TIẾN HÓA ---
    for it = 1:MaxIter
        for i = 1:NP
            % 1. Đột biến (Mutation - DE/rand/1)
            idxs = randperm(NP, 3);
            while any(idxs == i)
                idxs = randperm(NP, 3);
            end
            
            % Tạo Vector Đột biến
            Mutant = Pop(idxs(1), :) + F * (Pop(idxs(2), :) - Pop(idxs(3), :));
            
            % 2. Lai ghép (Crossover)
            Cross_Mask = rand(1, Dim) <= CR;
            Trial = Pop(i, :);
            Trial(Cross_Mask) = Mutant(Cross_Mask);
            
            % 3. Kiểm tra biên (Clamping)
            Trial = max(Trial, LowerBound);
            Trial = min(Trial, UpperBound);
            
            % 4. Chọn lọc (Selection)
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
    end
    
    % --- CHUYỂN ĐỔI KẾT QUẢ VỀ SỐ PHỨC ---
    w_real = BestSol(1:M);
    w_imag = BestSol(M+1:end);
    W_best = (w_real + 1i * w_imag).'; 
end

% --- HÀM MỤC TIÊU (COST FUNCTION) ---
function cost = CostFunction(x, Aq, PdM, alpha, M)
    % Tái tạo vector w từ biến thực x
    w_real = x(1:M);
    w_imag = x(M+1:end);
    w = (w_real + 1i * w_imag).';
    
    % Tính mẫu bức xạ tạo ra: |w^H * A|
    GeneratedPattern = abs(w' * Aq);
    
    % Chuẩn hóa (để so sánh hình dáng, không quan trọng độ lớn tuyệt đối)
    max_val = max(GeneratedPattern);
    if max_val > 0
         GeneratedPattern = GeneratedPattern / max_val;
    end
    
    % Tính sai số tại các điểm quan trọng (alpha)
    % Hàm mục tiêu: Tổng bình phương sai số (Least Squares Error)
    diff = GeneratedPattern(alpha) - PdM(alpha);
    cost = sum(abs(diff).^2); 
end