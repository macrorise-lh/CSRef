# coding=utf-8
# Copyright 2022 The SimREC Authors. All rights reserved.
# Copyright 2025 The CSRef Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class ProgressMeter(object):
    """
    A class to display and format training progress information.
    
    This class helps track and display the progress of training by showing
    the current epoch, batch number, and various metrics. It formats the
    output in a readable way, making it easy to monitor training progress.
    
    Attributes:
        fmtstr (str): Format string for epoch and batch display
        meters (list): List of AverageMeter objects to track metrics
        prefix (str): Prefix string to display before the progress information
    """
    def __init__(self, version, num_epochs, num_batches, meters, prefix=""):
        """
        Initialize the ProgressMeter.
        
        Args:
            version (str): Version identifier for the model or experiment
            num_epochs (int): Total number of epochs in the training
            num_batches (int): Total number of batches per epoch
            meters (list): List of AverageMeter objects to track metrics
            prefix (str, optional): Prefix string to display before progress info. Defaults to "".
        """
        self.fmtstr = self._get_epoch_batch_fmtstr(version, num_epochs, num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, epoch, batch):
        """
        Display the current training progress.
        
        This method formats and prints the current epoch, batch number,
        and the values of all tracked metrics.
        
        Args:
            epoch (int): Current epoch number
            batch (int): Current batch number within the epoch
        """
        entries = [self.prefix + self.fmtstr.format(epoch, batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_epoch_batch_fmtstr(self, version, num_epochs, num_batches):
        """
        Create a format string for displaying epoch and batch progress.
        
        This method generates a format string that ensures consistent
        spacing for epoch and batch numbers, regardless of their magnitude.
        
        Args:
            version (str): Version identifier for the model or experiment
            num_epochs (int): Total number of epochs in the training
            num_batches (int): Total number of batches per epoch
            
        Returns:
            str: Format string for epoch and batch display
        """
        # Calculate the number of digits needed for epoch and batch numbers
        num_digits_epoch = len(str(num_epochs // 1))
        num_digits_batch = len(str(num_batches // 1))
        # Create format strings for epoch and batch numbers
        epoch_fmt = '{:' + str(num_digits_epoch) + 'd}'
        batch_fmt = '{:' + str(num_digits_batch) + 'd}'
        # Combine into a single format string
        return '[' 'version: '+version+' '+ epoch_fmt + '/' + epoch_fmt.format(num_epochs) + ']' + '[' + batch_fmt + '/' + batch_fmt.format(
            num_batches) + ']'

class AverageMeter(object):
    """
    Computes and stores the average and current value of a metric.
    
    This class is useful for tracking various metrics during training,
    such as loss, accuracy, etc. It maintains running statistics including
    the current value, average, sum, and count of values.
    
    Attributes:
        name (str): Name of the metric being tracked
        fmt (str): Format string for displaying the metric value
        val (float): Current value of the metric
        avg (float): Average value of the metric
        sum (float): Sum of all metric values
        count (float): Count of metric values
        avg_reduce (float): Global average after reduction across processes
    """

    def __init__(self, name, fmt=':f'):
        """
        Initialize the AverageMeter.
        
        Args:
            name (str): Name of the metric being tracked
            fmt (str, optional): Format string for displaying the metric value. Defaults to ':f'.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Reset all statistics to zero.
        
        This method resets the current value, average, sum, count,
        and reduced average to their initial values.
        """
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
        self.avg_reduce = 0.

    def update(self, val, n=1):
        """
        Update the meter with a new value.
        
        This method updates the current value, sum, count, and average
        of the metric. It supports both single value updates (n=1)
        and batch updates (n>1). When n=-1, it sets the sum to the
        current value and count to 1, effectively replacing all
        previous values.
        
        Args:
            val (float): New value to update the meter with
            n (int, optional): Number of times the value should be counted.
                              Defaults to 1. Use n=-1 to replace all previous values.
        """
        self.val = val
        if n == -1:
            # Special case: replace all previous values with the current one
            self.sum = val
            self.count = 1
        else:
            # Normal case: accumulate the value
            self.sum += val * n
            self.count += n
        # Update the average
        self.avg = self.sum / self.count

    def update_reduce(self, val):
        """
        Update the reduced average value.
        
        This method is used to store the global average after reduction
        across all processes in a distributed training setup.
        
        Args:
            val (float): Global average value after reduction
        """
        self.avg_reduce = val

    def __str__(self):
        """
        Return a string representation of the meter.
        
        This method formats the meter's name and reduced average value
        into a readable string using the specified format.
        
        Returns:
            str: String representation of the meter
        """
        fmtstr = '{name} {avg_reduce' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

