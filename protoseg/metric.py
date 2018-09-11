from importlib import import_module


class Metric():
    global_step = 0

    def __init__(self, config, summarywriter=None):
        self.config = config
        self.summarywriter = summarywriter
        assert(config)

        self.metrices = []
        metrices = self.config.get('metrices')
        if metrices:
            print('___ loading metrices ___')
            for m in metrices:
                name = list(m.keys())[0]
                full_function = m[name]
                module_name, function_name = full_function.rsplit('.', 1)
                print(name, module_name, function_name)
                mod = import_module(module_name)
                met = getattr(mod, function_name)
                self.metrices.append(
                    {'name': name, 'function': met})

    def __call__(self, prediction, label, prefix = ''):
        self.global_step += 1
        for m in self.metrices:
            name = m['name']
            value = m['function'](prediction, label)
            print(name, "{0:.6f}".format(value))
            if self.summarywriter:
                self.summarywriter.add_scalar(
                    prefix + name, value, global_step=self.global_step)
