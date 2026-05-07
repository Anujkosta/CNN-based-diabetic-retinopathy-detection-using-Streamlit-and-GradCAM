import tensorflow as tf
import numpy as np

def find_last_conv_layer(model):
    backbone = None

    for layer in model.layers:
        if hasattr(layer, 'layers'):
            backbone = layer
            break

    if backbone is None:
        raise ValueError("No backbone found")

    conv_layers = [
        l for l in backbone.layers
        if isinstance(l, tf.keras.layers.Conv2D)
    ]

    if not conv_layers:
        raise ValueError("No Conv2D layers found")

    return conv_layers[-1]


def make_gradcam_heatmap(img_array, model, pred_index=None):

    last_conv_layer = find_last_conv_layer(model)

    conv_output_holder = []

    original_call = last_conv_layer.call

    def hooked_call(*args, **kwargs):
        output = original_call(*args, **kwargs)
        conv_output_holder.append(output)
        return output

    last_conv_layer.call = hooked_call

    try:
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            predictions = model(img_tensor, training=False)

            if pred_index is None:
                pred_index = tf.argmax(predictions[0])

            class_score = predictions[:, pred_index]

        conv_outputs = conv_output_holder[0]

    finally:
        last_conv_layer.call = original_call

    grads = tape.gradient(class_score, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()