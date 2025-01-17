import torch
shuffle_ratio = 0.3
device = "cpu"
batch_size = 1
attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
x = torch.arange(10, device=device).unsqueeze(0).expand(batch_size, -1)
x_shuffled = x.clone()


for b in range(batch_size):
    # -----------------------------
    # (1) attention_mask가 1인 위치(valid_indices)만 뽑기
    #     ex) valid_indices.shape = (num_valid,) 
    # -----------------------------
    valid_indices = attention_mask[b].nonzero(as_tuple=True)[0]  # 1인 곳의 인덱스
    num_valid = valid_indices.numel()

    if num_valid < 2:
        continue

    num_to_shuffle = int(num_valid * shuffle_ratio)
    if num_to_shuffle < 1:
        continue
    chosen_indices = valid_indices[torch.randperm(num_valid, device=device)[:num_to_shuffle]]

    # -----------------------------
    # (4) 뽑은 chosen_indices끼리만 다시 "랜덤 순서(perm)"로 재배열
    # -----------------------------
    perm_indices = torch.randperm(num_to_shuffle, device=device)
    original_values = x_shuffled[b, chosen_indices].clone()  # 임시 복사
    x_shuffled[b, chosen_indices] = original_values[perm_indices]
    
print(x, x_shuffled)