import tensorflow as tf  
from model import build_model
from metrics.segmentation_metrics import dice_coeff, bce_dice_loss, IoU, zero_IoU, dice_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
import os 
from sklearn.model_selection import train_test_splits
from callbacks.callbacks import get_callbacks
from dataloader.dataloader import build_augmenter, build_dataset, build_decoder
import yaml
from yaml.loader import SafeLoader

def main(config_file):
    
    with open(config_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
        print("Parameter for Training:")
        print(data)
    
    img_size = data['Data']['img_size']
    route = data['Data']['route']
    X_path = data['Data']['X_path']
    Y_path = data['Data']['Y_path']
    
    valid_size = data['Hyperparameter']['valid_size']
    test_size = data['Hyperparameter']['test_size']
    SEED = data['Hyperparameter']['SEED']
    BATCH_SIZE = data['Hyperparameter']['BATCH_SIZE']
    epochs = data['Hyperparameter']['epochs']
    max_lr = data['Hyperparameter']['max_lr']
    min_lr = data['Hyperparameter']['min_lr']
    save_weights_only = data['Hyperparameter']['save_weights_only']
    
    save_path = data['Model']['save_path']
    
    
    print("BUILD MODEL:")
    
    model = build_model(img_size)
    model.summary()
    
    get_custom_objects().update({"dice": dice_loss})
    model.compile(optimizer=Adam(lr=1e-3),
                loss='dice',
                metrics=[dice_coeff, bce_dice_loss, IoU, zero_IoU])

    print("LOAD DATA:")
    X_full = sorted(os.listdir(f'{route}/images'))
    Y_full = sorted(os.listdir(f'{route}/masks'))

    print(len(X_full))

    X_train, X_valid = train_test_splits(X_full, test_size=valid_size, random_state=SEED)
    Y_train, Y_valid = train_test_splits(Y_full, test_size=valid_size, random_state=SEED)

    X_train, X_test = train_test_splits(X_train, test_size=test_size, random_state=SEED)
    Y_train, Y_test = train_test_splits(Y_train, test_size=test_size, random_state=SEED)

    print("N Train:", len(X_train))
    print("N Valid:", len(X_valid))
    print("N test:", len(X_test))
    
    train_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
    train_dataset = build_dataset(X_train, Y_train, bsize=BATCH_SIZE, decode_fn=train_decoder, 
                                augmentAdv=False, augment=False, augmentAdvSeg=True)

    valid_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
    valid_dataset = build_dataset(X_valid, Y_valid, bsize=BATCH_SIZE, decode_fn=train_decoder, 
                                augmentAdv=False, augment=False, repeat=False, shuffle=False,
                                augmentAdvSeg=False)

    test_decoder = build_decoder(with_labels=True, target_size=(img_size, img_size), ext='jpg', segment=True, ext2='jpg')
    test_dataset = build_dataset(X_test, Y_test, bsize=BATCH_SIZE, decode_fn=test_decoder, 
                                augmentAdv=False, augment=False, repeat=False, shuffle=False,
                                augmentAdvSeg=False)
    
    callbacks = get_callbacks(monitor = 'val_loss', mode = 'min', save_path = save_path, max_lr = max_lr
                          , min_lr = min_lr , cycle_epoch = 1000, save_weights_only = save_weights_only)
    
    steps_per_epoch = len(X_train) // BATCH_SIZE
    
    print("START TRAINING:")
    
    his = model.fit(train_dataset, 
                epochs=epochs,
                verbose=1,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                validation_data=valid_dataset)
    
if __name__ == "__main__":
    config_file = "./config/train_config.yaml"
    main(config_file)