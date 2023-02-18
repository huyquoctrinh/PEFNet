from backbone.basic_backbone import EfficientNetV2B0, ConvnextSmall
from layers.layers import merge, bn_act, decode, conv_pos_emb
from tensorflow.keras.layers import Conv2D
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

def build_model(img_size, num_classes = 1, backbone = 'efficientnetv2b0'):
    
    if backbone == 'efficientnetv2-b0':
        
        backbone, layer_names = EfficientNetV2B0(img_size)
        
    else:
        
        backbone, layer_names = ConvnextSmall(img_size)
    
    layers = [backbone.get_layer(x).output for x in layer_names]

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    x = layers[0]

    for i, layer in enumerate(layers[1:]):
        x = decode(x, layer.shape[channel_axis])
        x = merge([x, layer], layer.shape[channel_axis])
        x = conv_pos_emb(x, layer.shape[channel_axis])

    filters = x.shape[channel_axis] // 2

    x = decode(x, filters)
    x = conv_pos_emb(x, filters)

    x = Conv2D(num_classes, kernel_size=1, padding='same', activation='sigmoid')(x)

    model = Model(backbone.input, x)
    
    return model
