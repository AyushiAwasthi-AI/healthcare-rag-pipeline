def __init__(self, chunk_size: int =512, chunk_overlap: int =None):
    self.chunk_size = chunk_size
#if overlap not specificed - always 25% of chunk size
    self.chunk_overkap = chunk_overlap or (chunk_size//4)