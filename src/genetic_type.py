class AttrEvent():

    def __init__(self, date, time, dev, attr, value):
        # Event in the format of [date, time, dev, dev_attr, value]
        self.date = date; self.time = time; self.dev = dev; self.attr = attr; self.value = value
    
    def __str__(self) -> str:
        return ' '.join([self.date, self.time, self.dev, self.attr, self.value])

class DevAttribute():

    def __init__(self, attr_name=None, attr_index=None, lag=0):
        self.index = attr_index
        self.name = attr_name
        self.lag = lag