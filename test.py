from tensorflow.keras.layers.models import load_model
from backbone.basic_backbone import EfficientNetV2B0, ConvnextSmall
import cv2 
from model import build_model
import matplotlib.pyplot as plt 
import tensorflow as tf 
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder
import os  
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss
import yaml
from yaml.loader import SafeLoader

def inference_single_image(model_path, img_size, img_path,  model_name = 'efficientnetv2-b0'):
    
    print("LOAD MODEL:")
    model = build_model(img_size, backbone = model_name)
    model.load_weights(model_path)
    model.summary()
    
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_size, img_size))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    mask = model.predict(img_array)
    
    cv2.imwrite("result.png", mask)
    
    return mask 

def benchmark_batch_images(route, model_path, img_size, BATCH_SIZE, model_name = 'efficientnetv2-b0'):
    
    print("LOAD MODEL:")
    model = build_model(img_size, backbone = model_name)
    model.load_weights(model_path)
    model.summary()
    
    X_test = sorted(os.listdir(f'{route}/images'))
    Y_test = sorted(os.listdir(f'{route}/masks'))
    
    test_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
    test_dataset = build_dataset(X_test, Y_test, bsize=BATCH_SIZE, decode_fn=test_decoder, 
                                augmentAdv=False, augment=False, repeat=False, shuffle=False,
                                augmentAdvSeg=False)
    
    
    model.compile(metrics = [dice_coeff, bce_dice_loss, IoU, zero_IoU])
    
    dice_coeff, dice_loss, iou, zero_iou = model.evaluate(test_dataset)
    
    print("Dice Coeff:", dice_coeff)
    print("Dice Loss:", dice_loss)
    print("IOU:", iou)
    print("Zero IOU:", zero_iou)
    
    return dice_coeff, dice_loss, iou, zero_iou

if __name__ == "__main__":
    
    benchmark_config = 'benchmark_config.yaml'
    with open(benchmark_config) as f:
        data = yaml.load(f, Loader=SafeLoader)
        print("Parameter for Training:")
        
    benchmark_batch_images(data['route'], data['model_path'], 
                           data['img_size'], data['BATCH_SIZE'], 
                           data['model_name'])
    