'''
This module contains the converter factory class for creating converter objects.
'''

from base_converter import Converter
from ls_converter import LeastSquaresConverter
from bp_converter import BPConverter
from mlp_ust_converter import MLPUnsupervisedConverter
from mlp_st_converter import MLPSupervisedConverter
from random_converter import RandomConverter

converter_classes = {
    "ls": LeastSquaresConverter,
    "bp": BPConverter,
    "mlp_ust": MLPUnsupervisedConverter,
    "mlp_st": MLPSupervisedConverter,
    "random": RandomConverter,
}

class ConverterFactory:
    '''
    This class contains the converter factory class for creating converter objects.
    '''
    def __init__(self):
        '''
        Initialize the converter factory class.
        '''
        self.converter_classes = converter_classes

    def create_converter(self, converter_type, converter_config=None, *args, **kwargs):
        if converter_type not in self.converter_classes:
            raise ValueError(f"Invalid converter type: {converter_type}")
        converter_class = self.converter_classes[converter_type]
        if converter_config is not None:
            for key, value in converter_config.items():
                kwargs[key] = value
        return converter_class(*args, **kwargs)