from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class WeightedAverage(Layer):

    def __init__(self, **kwargs):
        super(WeightedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, input_shape[-2]),
                                      initializer='uniform',
                                      trainable=True)
        super(WeightedAverage, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        weighted_average = K.dot(self.kernel, x)
        return weighted_average
        return K.squeeze(weighted_average, 1)

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        del output_shape[-2]
        return tuple(output_shape)



if __name__ == '__main__':
    def test():
        #Test the layer
        import keras
        from keras.models import Model
        import keras.layers

        sent1 = [                             #Satz1
            [[0, 1, 2, 3, 4, 5],             #Wort1
             [100, 101, 102, 103, 104, 105],
             [5, 4, 3, 2, 1, 0]]

           ,[[1, 2, 3, 4, 5, 6],            #Wort2
            [4, 5, 6, 7, 8, 9],
            [6, 7, 8, 9, 0, 1]]
        ]

        sent2 = [                             #Satz1
            [[5, 5, 5, 5, 5, 5],             #Wort1
             [10, 11, 12, 13, 14, 10],
             [5, 6, 7, 8, 2, 1]]

           ,[[33, 22, 333, 444, 555, 666],            #Wort2
            [44, 55, 66, 77, 88, 99],
            [66, 77, 88, 99, 00, 11]]
        ]

        sent3 = sent1
        sent4 = sent2



        inputs = keras.layers.Input(shape=(None, 3, 6))
        wa = keras.layers.TimeDistributed(WeightedAverage())(inputs)
        predictions = keras.layers.Dense(10)(wa)
        model = Model(inputs=inputs, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.summary()

        data = np.array([sent1, sent2, sent3, sent4])
        y = model.predict(x=[data])

        print("data.shape", data.shape)
        print("Expected y.shape: 2x1x6")
        print("y.shape", y.shape)
        print("y", y )
        print("weights", model.get_weights())

    test()