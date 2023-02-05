# class for cnn layers. base class that every layer inherits from
class Layer:
    def forward(self, input):
        # input.
        # returns output
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        # output_error -> part_L / part_output
        # learning_rate -> learning rate
        # calculates kernel_error and bias_error
        # returns input_error
        raise NotImplementedError
