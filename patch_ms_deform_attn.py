#!/usr/bin/env python3
import os

# Fix ms_deform_attn.h
with open('ops/src/ms_deform_attn.h', 'r') as f:
    content = f.read()

# Replace deprecated type() calls with device().is_cuda()
content = content.replace('value.type().is_cuda()', 'value.device().is_cuda()')

with open('ops/src/ms_deform_attn.h', 'w') as f:
    f.write(content)

# Fix ms_deform_attn_cuda.cu
with open('ops/src/cuda/ms_deform_attn_cuda.cu', 'r') as f:
    content = f.read()

# Replace AT_DISPATCH_FLOATING_TYPES with newer version
content = content.replace('AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda"', 
                         'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_forward_cuda"')
content = content.replace('AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda"', 
                         'AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "ms_deform_attn_backward_cuda"')

with open('ops/src/cuda/ms_deform_attn_cuda.cu', 'w') as f:
    f.write(content)
