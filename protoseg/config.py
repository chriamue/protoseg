import yaml


class Config():

    keys = []
    current = -1

    default = {'datapath': 'data/',
               'backend': 'gluoncv_backend', 'backbone': 'resnet50',
               'batch_size': 1, 'learn_rate': 1.0, 'epochs': 1, 'dropout': 0.5, # hyperparameter
               'optimizer': 'sgd',
               'loss_function': 'default', 'loss_function_parameters': {},
               'pretrained': False, 'summarysteps': 100, 'classes': 2,
               'width': 480, 'height': 480,
               'orig_width': 512, 'orig_height': 512,
               'gray_img': False, 'gray_mask': False,
               'color_img': False, 'color_mask': False,
               'flip': False, 'horizontal_flip': True,
               'rotation_degree': 0,
               'horizontal_shift': 0, 'vertical_shift': 0,
               'noise_amount': 0, 'noise_chance': 0,
               'min_val': 0, 'max_val': 255,  # pixel values
               'min_bright': -20, 'max_bright': +30,  # brightness
               'zoom_in': 0, 'zoom_out': 0,  # zoom
               'img_augmentation': [],'shape_augmentation': [], 'filters': [],
               'hyperparamopt': []
               }

    def __init__(self, configs={}):
        """
        """
        if isinstance(configs, dict):
            self.configs = configs
        else:
            with open(configs) as file:
                self.configs = yaml.load(file)
                self.filename = configs
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
