from __future__ import absolute_import
from __future__ import print_function

import os
import json

from keras.callbacks import Callback

try:
    import requests
except ImportError:
    requests = None


class SlackLogger(Callback):

    def __init__(self, logger_name='Slack Logger :memo:', api_key='',
                 header=None, epochs=None, debug=False):

        self.logger_name = logger_name
        self.api_key = api_key
        self.header = header
        self.epochs = epochs
        self.debug = debug
        self.notify(f'Learning launched! My name is {logger_name} :fire:')

    def on_epoch_begin(self, epoch, logs=None):
        self.seen = 0
        self.totals = {}

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size

        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        message = ''

        if self.header:
            message += f'{self.header}\n'

        message += f'logger_name: {self.logger_name}\n\n'
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    logs[k] = self.totals[k] / self.seen

            if self.epochs:
                message += f'epoch: {epoch} / {self.epochs}\n'

            else:
                message += f'epoch: {epoch}\n'

            for k, v in logs.items():
                message += f'{k}: {v}\n'

            if self.epochs:
                if epoch == self.epochs:
                    message += '\nTerminated!'

            self.notify(message)

    def notify(self, message):
        if self.debug:
            print(message)
            return 0

        data = {'text': message}
        data = json.dumps(data)

        base_url = 'https://hooks.slack.com/services'
        cmd = f'curl -X POST -H \'Content-type: application/json\' --data \'{data}\' {base_url}/{self.api_key}'  # NOQA
        os.system(cmd)
