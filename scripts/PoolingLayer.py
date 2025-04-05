from typing import Tuple, List
import numpy as np

class PoolingLayer:
    def __init__(self, pooling_type: str="MAX", pool_size: Tuple[int, int]=(2, 2)):
        assert pooling_type in ['MAX', 'AVERAGE'], "Pooling type must be 'MAX' or 'AVERAGE'"
        self.pooling_type = pooling_type
        self.pool_size = pool_size
        self.pool_fn = self._max_pool if pooling_type == 'MAX' else self._average_pool

    def _max_pool(self, pool_data: np.ndarray) -> np.ndarray:
        # return maximum value in the pool
        return np.max(pool_data, axis=(0, 1))
    
    def _average_pool(self, pool_data: np.ndarray) -> np.ndarray:
        # return average value in the pool
        return np.mean(pool_data, axis=(0, 1))
    
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        '''
        Apply pooling operation over the input data.
        
        Args:
            input_data (np.ndarray): Input feature map (H, W, C)
        
        Returns:
            np.ndarray: Pooled output (H//pool_H, W//pool_W, C)
        '''
        input_h, input_w, channels = input_data.shape
        pool_h, pool_w = self.pool_size

        output_h = input_h // pool_h
        output_w = input_w // pool_w
        output = np.zeros((output_h, output_w, channels))

        for h in range(output_h):
            for w in range(output_w):
                h_start = h * pool_h
                h_end = h_start + pool_h
                w_start = w * pool_w
                w_end = w_start + pool_w

                pool_region = input_data[h_start:h_end, w_start:w_end, :]
                output[h, w, :] = self.pool_fn(pool_region)

        return output


# test poolinglayer
if __name__ == "__main__":
    # Create a random input tensor with shape (H, W, C)
    input_tensor = np.random.rand(4, 4, 3)  # Example input tensor

    # Create a PoolingLayer instance
    pooling_layer = PoolingLayer(pooling_type='MAX', pool_size=(2, 2))

    # Apply the pooling layer to the input tensor
    output_tensor = pooling_layer.forward(input_tensor)

    print("Input Tensor:")
    print(input_tensor)
    print("\nOutput Tensor after Max Pooling:")
    print(output_tensor)
    print("\nShape of Output Tensor:")