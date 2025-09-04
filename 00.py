import torch

scala = torch.tensor(7)

print(f"Scala tensor: {scala}")

print(f"Scala ndim: {scala.ndim}")

print(f"Scala Item: {scala.item()}")

vector = torch.tensor([7, 7])

print(f"Vector tensor: {vector}")

print(f"Vector ndim: {vector.ndim}")

print(f"Vector shape: {vector.shape}")

matrix = torch.tensor(
    [
        [7,8],
        [9,10]
    ]
)

print(f"Matrix value: {matrix}")

print(f"Matrix ndim: {matrix.ndim}")

print(f"Matrix shape: {matrix.shape}")

tensor = torch.tensor(
    [
        [
            [1,2,3],
            [3,6,9],
            [2,4,5]
        ]
    ]
)

print(f"Tensor value: {tensor}")

print(f"Tensor ndim: {tensor.ndim}")

print(f"Tensor shape: {tensor.shape}")

random_tensor = torch.rand(size=(3,4))
print(f"Random tensor: {random_tensor}")
print(f"Random tensor shape: {random_tensor.shape}. Random tensor ndim: {random_tensor.ndim}")

tensor_zero = torch.zeros(size=(3,4))
print(f"Tensor zero: {tensor_zero}. Tensor zero dtype: {tensor_zero.dtype}")

tensor_one = torch.ones(size=(3,4))
print(f"Tensor one: {tensor_one}. Tensor one dtype: {tensor_one.dtype}")

zero_to_ten = torch.arange(start=0,end=10,step=1)
print(f"Zero to ten: {zero_to_ten}")

ten_zeros = torch.zeros_like(input=zero_to_ten)
print(f"Ten zeros: {ten_zeros}")

# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device='cuda', # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

# float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device
print(f"Float 32 tensor shape: {float_32_tensor.shape}")
print(f"Float 32 tensor dtype: {float_32_tensor.dtype}")
print(f"Float 32 tensor device: {float_32_tensor.device}")


