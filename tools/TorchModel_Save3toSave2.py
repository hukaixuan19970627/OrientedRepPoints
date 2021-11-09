import torch

# torch ≥ 1.5 environment
torch3_checkpoint = 'work_dirs/swin_tiny_patch4_window7_224_torch3.pth'  
torch2_checkpoint = 'work_dirs/swin_tiny_patch4_window7_224_torch2.pth'  

checkpoint = torch.load(f=torch3_checkpoint)
torch.save(checkpoint, f=torch2_checkpoint, _use_new_zipfile_serialization=False)
print('The pretrained model has been turned into torch read version2.')