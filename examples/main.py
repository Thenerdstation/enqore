import tensornetwork
from tensornetwork import TensorNetwork, Node
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


def add_x(net: TensorNetwork) -> Node:
    _x_matrix = np.array(
        [[0, 1],
         [1, 0]], dtype=np.float32
    )
    x_node = net.add_node(_x_matrix)
    return x_node


if __name__ == '__main__':
    net = tensornetwork.TensorNetwork()
    x = add_x(net)
    test = net.add_node(np.array([1.0, 0.0], dtype=np.float32))
    edge = net.connect(test[0], x[0])
    ans = net.contract(edge)
    print(ans.tensor)
