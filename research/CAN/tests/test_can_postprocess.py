import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import math
import numpy as np
from mindocr import build_postprocess


def test_can_postprocess():
    """
        This test case is used to test whether the model post-processing can be loaded correctly
    """
    name = "CANLabelDecode"
    character_dict_path="/home/nginx/work/zhangjunlong/mindocr_mm/mindocr/utils/dict/latex_symbol_dict.txt"
    config = dict(name=name, character_dict_path=character_dict_path)
    data_decoder = build_postprocess(config)


if __name__=="__main__":
    test_can_postprocess()
