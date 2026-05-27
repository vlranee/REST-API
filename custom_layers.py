import tensorflow as tf
@tf.keras.utils.register_keras_serializable(
    package='Custom',
    name='TextPreprocessingLayer'
)
class TextPreprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, max_len, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.max_len = max_len

        self.vectorizer = (
            tf.keras.layers.TextVectorization(
                max_tokens=vocab_size,
                output_mode='int',
                output_sequence_length=max_len,
                standardize='lower_and_strip_punctuation'
            )
        )

    def build(self, input_shape):
        self.vectorizer.build(
            input_shape
        )

        super().build(
            input_shape
        )


    def call(self, inputs):
        return self.vectorizer(
            inputs
        )


    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'max_len': self.max_len
        })

        return config

@tf.keras.utils.register_keras_serializable(
    package='Custom',
    name='WeightedBinaryCrossentropy'
)
class WeightedBinaryCrossentropy(
    tf.keras.losses.Loss
):

    def __init__(
        self,
        weight_0=1.0,
        weight_1=1.2,
        label_smoothing=0.0,
        confidence_penalty=0.3,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.weight_0 = weight_0
        self.weight_1 = weight_1
        self.label_smoothing = label_smoothing
        self.confidence_penalty = confidence_penalty


    def call(
        self,
        y_true,
        y_pred
    ):

        bce = tf.keras.backend.binary_crossentropy(
            y_true,
            y_pred
        )

        weights = (
            y_true*self.weight_1+
            (1-y_true)*self.weight_0
        )

        return tf.reduce_mean(
            bce*weights
        )


    def get_config(self):
        config = super().get_config()
        config.update({
            "weight_0":
            self.weight_0,

            "weight_1":
            self.weight_1,

            "label_smoothing":
            self.label_smoothing,

            "confidence_penalty":
            self.confidence_penalty
        })
        return config