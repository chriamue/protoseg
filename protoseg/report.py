
import os
import numpy as np
import cv2
import json
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator as ea

from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
sns.set(style="darkgrid")
sns.set_context("paper")
from matplotlib.backends.backend_pdf import PdfPages

class Report():

    def __init__(self, configs, resultspath='results/'):
        self.configs = configs
        self.resultspath = resultspath
        assert(configs)

    # source: https://github.com/JamesChuanggg/Tensorboard2Seaborn/blob/master/beautify.py
    def plot(self, log_path, tag='loss', smooth_space=100, color_code='#4169E1'):
        ''' beautify tf log
            Use better library (seaborn) to plot tf event file'''

        acc = ea.EventAccumulator(log_path)
        acc.Reload()

        # only support scalar now
        scalar_list = acc.Tags()['scalars']

        x_list = []
        y_list = []
        x_list_raw = []
        y_list_raw = []
        for tag_ in scalar_list:
            if tag_ != tag:
                continue
            x = [int(s.step) for s in acc.Scalars(tag)]
            y = [s.value for s in acc.Scalars(tag)]

            # smooth curve
            x_ = []
            y_ = []
            for i in range(0, len(x), smooth_space):
                x_.append(x[i])
                y_.append(sum(y[i:i+smooth_space]) / float(smooth_space))
            x_.append(x[-1])
            y_.append(y[-1])
            x_list = x_
            y_list = y_

            # raw curve
            x_list_raw = x
            y_list_raw = y

        fig, ax = plt.subplots()
        plt.title(tag)
        plt.plot(x_list_raw, y_list_raw,
                     color=colors.to_rgba(color_code, alpha=0.4))
        plt.plot(x_list, y_list, color=color_code, linewidth=1.5)
        fig.canvas.draw()
        return fig, np.array(fig.canvas.renderer._renderer)

    def image(self, log_path, tag='loss'):
        acc = ea.EventAccumulator(log_path)
        acc.Reload()
        image_list = acc.Images(tag = tag)
        with tf.Session() as sess:
            img = tf.image.decode_image(image_list[-1].encoded_image_string)
            npimg = img.eval(session=sess)
        return npimg


    def generate(self):
        pp = PdfPages(os.path.join(self.resultspath, os.path.basename(self.configs.filename) + '.pdf'))
        for run in self.configs:
            resultpath = os.path.join(self.resultspath, run)
            fig, img = self.plot(resultpath, tag="loss")
            plt.text(0.05,0.95,run, transform=fig.transFigure, size=24)
            pp.savefig(fig)
            cv2.imwrite(resultpath+'/loss.png', img)
            config = self.configs.get()
            for metric in config['metrices']:
                name = list(metric.keys())[0]
                fig, img = self.plot(resultpath, tag=name)
                pp.savefig(fig)
                cv2.imwrite(resultpath+'/'+name+'.png', img)
        pp.close()

    def hyperparamopt(self, config, hyperparamoptimizer, resultpath):
        filename = os.path.join(resultpath, 'trials.csv')
        df = pd.DataFrame(data=hyperparamoptimizer.trials.results)
        df = df.set_index('loss')
        df.to_csv(filename)
        pp = PdfPages(os.path.join(resultpath, 'paramopt.pdf'))
        
        for result in hyperparamoptimizer.trials.results:
            trial = result['trial']
            _, loss = self.plot(log_path = resultpath, tag='trial'+str(trial)+'_loss')
            val_image = self.image(log_path = resultpath, tag='trial'+str(trial)+'_val_image')
            val_mask = self.image(log_path = resultpath, tag='trial'+str(trial)+'_val_mask')
            val_predicted = self.image(log_path = resultpath, tag='trial'+str(trial)+'_val_predicted')
            fig=plt.figure()
            fig.add_subplot(2, 4, 1)
            plt.imshow(loss)
            fig.add_subplot(2, 4, 2)
            plt.imshow(val_image)
            fig.add_subplot(2, 4, 3)
            plt.imshow(val_mask)
            fig.add_subplot(2, 4, 4)
            plt.imshow(val_predicted)
            plt.text(0.05,0.95, 'trial ' + str(trial), transform=fig.transFigure, size=24)
            for i, m in enumerate(config['metrices']):
                name = list(m.keys())[0]
                _, metric = self.plot(log_path = resultpath, tag='trial'+str(trial)+'_'+name)
                fig.add_subplot(2, len(config['metrices']), len(config['metrices']) + i+1)
                plt.imshow(metric)
            pp.attach_note(result['params'])
            pp.savefig(fig)
        pp.close()


