"""Process-lifetime mapped allocations for a one-shot experimental worker only.
No explicit free: only the driver/process teardown releases the mappings after all queues stop.
Never use with hot reload, in-process model teardown/restart, forked inference or multiple models.
"""
import ctypes as C
import ctypes.util
import torch
P=C.c_void_p
lib=C.CDLL(ctypes.util.find_library('musart'));owners=[]
for name,args in [('musaHostAlloc',[C.POINTER(P),C.c_size_t,C.c_uint]),('musaHostGetDevicePointer',[C.POINTER(P),P,C.c_uint])]:
    f=getattr(lib,name);f.argtypes=args;f.restype=C.c_int

def check(n):
    if n:raise RuntimeError(f'mapped-bank MUSA status {n}')
class Bank:
    def __init__(self,shape,dtype,device):
        self.host=P();d=P();self.width=shape[1]*shape[2]*torch.empty((),dtype=dtype,device='cpu').element_size();n=shape[0]*self.width
        check(lib.musaHostAlloc(C.byref(self.host),n,2));check(lib.musaHostGetDevicePointer(C.byref(d),self.host,0));s=torch._C._construct_storage_from_data_pointer(d.value,device,n);self.tensor=torch.empty(0,dtype=dtype,device=device).set_(s,0,shape);assert self.tensor.data_ptr()==d.value;owners.append(self)
    def slot(self,i):return memoryview((C.c_uint8*self.width).from_address(self.host.value+i*self.width)).cast('B')
