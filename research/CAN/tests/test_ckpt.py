import mindspore as ms   
import numpy as np  

def test_ckpt():
    """
        Used to verify that the weight file can be loaded correctly.
        And according to the need, print and check the weight parameters.
        You need to change the local directory of the weight file.
    """

    # Load the checkpoint file  
    checkpoint_path = '/home/ma-user/work/mindocr_mm/tests/ut/can_params_from_paddle.ckpt'  
    param_dict = ms.load_checkpoint(checkpoint_path)  

    # Iterate through the parameter dictionary and print the parameter names and values 
    for param_name, param_value in param_dict.items():

        # Convert Tensor to a NumPy array
        param_value_numpy = param_value.asnumpy()  

        print(f"Parameter Name: {param_name}")  
        print(f"Parameter Shape: {param_value_numpy.shape}")  
        print(f"Parameter Values (first few elements): {param_value_numpy[:5]}")
        print("-" * 40)


if __name__=="__main__":
    test_ckpt()
