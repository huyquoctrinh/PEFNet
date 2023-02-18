from keras_cv_attention_models import convnext, efficientnet

def ConvnextSmall(im_size):
    
    backbone = convnext.ConvNeXtSmall(input_shape=(im_size, im_size, 3), num_classes=0, 
                                pretrained='imagenet21k-ft1k')

    layer_names = [
        'stack4_block3_output',
        'stack3_block9_output',
        'stack2_block3_output',
        'stack1_block3_output',
    ]
    
    return backbone, layer_names

def EfficientNetV2B0(im_size):
    
    backbone = efficientnet.EfficientNetV2B0(input_shape=(im_size, im_size, 3), num_classes=0, 
                                            pretrained='imagenet21k', include_preprocessing=True)
    
    layer_names = ['post_swish', 
                    'stack_4_block4_output', 
                    'stack_2_block1_output', 
                    'stack_1_block1_output', 
                    'stack_0_block0_output']
    
    return backbone, layer_names
