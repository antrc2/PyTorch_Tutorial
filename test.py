import torch

t = torch.rand(1,3,1,5)

print(t)
print(t.shape)           # (1, 3, 1, 5)
print(t.squeeze().shape) # (3, 5)     # bỏ hết dim=1
print(t.squeeze(0).shape) # (3, 1, 5) # chỉ bỏ dim=0
print(t.squeeze(1).shape) # (1, 3, 1, 5) vì dim=1=3 nên không bỏ

print(t.squeeze())

t = torch.tensor([10, 20, 30])   # shape (3,)

print(t.unsqueeze(0).shape)  # (1, 3)  => thêm chiều ở đầu (batch size = 1)
print(t.unsqueeze(1).shape)  # (3, 1)  => thêm chiều ở cuối
print(t.unsqueeze(1))