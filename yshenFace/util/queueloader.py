class queueloader:
    def __init__(self, queue, batch_size=1):
        self.queue = queue
        #every time we deal with no more than batch_size images
        self.batch_size = batch_size
        self.data = None
        self.curindex = 0

    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
        
    def next(self):
        if not self.queue.empty() and self.curindex < self.batch_size:
            self.get_batch()
            return self.data
        else:
            self.curindex = 0    #reset
            raise StopIteration

    def get_batch(self):
        img = self.queue.get()
        #assert len(img) == 1, "Single batch only"
        self.data = img
        self.curindex += 1


