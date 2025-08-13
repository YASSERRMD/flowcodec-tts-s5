import os, json, random
from torch.utils.data import Sampler
class BucketByLengthSampler(Sampler):
    def __init__(self, index_json, batch_size=1, max_frames=900, shuffle=True):
        data = json.load(open(index_json)); self.buckets = {}
        for it in data:
            f = min(int(it['frames']), max_frames); key = (f//100)*100
            self.buckets.setdefault(key, []).append(it['path'])
        self.keys = sorted(self.buckets.keys()); self.batch_size=batch_size; self.shuffle=shuffle
    def __iter__(self):
        keys = self.keys[:]
        if self.shuffle: random.shuffle(keys)
        for k in keys:
            items = self.buckets[k][:]
            if self.shuffle: random.shuffle(items)
            for i in range(0, len(items), self.batch_size):
                yield items[i:i+self.batch_size]
    def __len__(self):
        return sum((len(v)+self.batch_size-1)//self.batch_size for v in self.buckets.values())
