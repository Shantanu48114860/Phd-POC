import numpy as np
import torch

lt = []
batch1 = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
batch2 = torch.tensor([[2, 3, 4, 4], [2, 3, 4, 4]])
ten2 = torch.cat((batch1, batch2), dim=0)
print(ten2)
print(ten2.shape)

ten3 = torch.tensor(())
for i in range(2):
    batch = torch.randn(2, 4)
    print(batch)
    ten3 = torch.cat((ten3, batch), dim=0)
print("####")
print(ten3)
print("####")

# arr = []
# arr.extend(np.array([[1., 2., 3., 4.], [1., 2., 3., 4.]], dtype=float))
# arr.extend(np.array([[2., 3., 4., 4., 6.], [2, 3., 4., 4., 6.]], dtype=float))
# ten1 = np.array(arr)
# print(arr)
# print(ten1.shape)

ten1 = torch.tensor([[1, 2, 3], [1, 2, 3]])
print(ten1)
