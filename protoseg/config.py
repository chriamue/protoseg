import yaml

class Config():

    keys = []
    current = -1

    def __init__(self, configs={}):
        """
        """
        if isinstance(configs, dict):
            self.configs = configs
        else:
            with open(configs) as file:
                self.configs = yaml.load(file)
        self.keys = list(self.configs.keys())

    def __iter__(self):
        self.current = -1
        return self

    def __next__(self):
        if self.current + 1 >= len(self):
            raise StopIteration
        else:
            self.current += 1
            return self.keys[self.current]

    def __getitem__(self, index):
        return self.configs[self.keys[index]]

    def __len__(self):
        return len(self.keys)

    def save(self, path):
        """
        save current config to given path as yaml file
        """
        with open(path, 'w') as outfile:
            yaml.dump({self.keys[self.current]: self[self.current]}, outfile, default_flow_style=False)

    def get(self, key=None):
        if key is None:
            key = self.keys[self.current]
        return self[self.keys.index(key)]