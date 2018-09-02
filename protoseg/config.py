import yaml


class Config():

    keys = []
    current = -1

    default = {'backend': 'gluoncv_backend', 'backbone': 'resnet50',
               'batch_size': 1, 'learn_rate': 1.0, 'epochs': 1,  # hyperparameter
               'pretrained': False,
               'width': 480, 'height': 480,
               'flip': False, 'horizontal_flip': True,
               'rotation_degree': 0,
               'horizontal_shift': 0, 'vertical_shift': 0,
               'noise_amount': 0, 'noise_chance': 0,
               'min_val': 0, 'max_val': 255,  # pixel values
               'min_bright': -20, 'max_bright': +30,  # brightness
               'zoom_in': 0, 'zoom_out': 0  # zoom
               }

    def __init__(self, configs={}):
        """
        """
        if isinstance(configs, dict):
            self.configs = configs
        else:
            with open(configs) as file:
                self.configs = yaml.load(file)
        self.keys = list(self.configs.keys())
        for run in self:
            self.fill_missing(self.get())

    def fill_missing(self, config):
        for key in self.default.keys():
            if config.get(key) is None:
                config[key] = self.default[key]

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
            yaml.dump({self.keys[self.current]: self[self.current]},
                      outfile, default_flow_style=False)

    def get(self, key=None):
        if key is None:
            key = self.keys[self.current]
        return self[self.keys.index(key)]

    def current_run(self):
        return self.keys[self.current]
