import keras
from keras.models import Model, load_model
from keras.regularizers import l2
import warnings
import tensorflow as tf

class Multi_gpu_Checkpoint(keras.callbacks.ModelCheckpoint):
    def __init__(self, original_model, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(Multi_gpu_Checkpoint, self).__init__()
        self.original_model = original_model

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.original_model.save_weights(
                                filepath, overwrite=True)
                        else:
                            self.original_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' %
                          (epoch + 1, filepath))
                if self.save_weights_only:
                    self.original_model.save_weights(filepath, overwrite=True)
                else:
                    self.original_model.save(filepath, overwrite=True)


def custom_loss(y_true, y_pred):
    # custom loss to calculate the loss
    # y_pred is some batch*size*size*point_num image
    # y_true is same size
    # vis = 0 or 1, 0 means unvisable
    vis = keras.backend.max(y_true, axis=-2, keepdims=True)
    vis = keras.backend.max(vis, axis=-3, keepdims=True)
    y_true = y_true * vis
    y_pred = y_pred * vis
    return keras.losses.binary_crossentropy(y_true,y_pred)


class CPN(object):
    def __init__(self, point_num = 24, weight_decay=0., gpu_num=1, load_model=None):
        pass
        self.point_num = point_num
        self.gpu_num = gpu_num
        self.i = 1
        if load_model == None:
            self.model = self.get_model(weight_decay=weight_decay)
        else:
            self.model = keras.models.load_model(load_model,{'custom_loss':custom_loss})
    
    def res_bottleneck(self, input_tensor, weight_decay):
        tmp = keras.layers.Conv2D(128, kernel_size=(1, 1), padding='same', name='conv_2d_bottle_%d' %
                                  self.i, kernel_regularizer=l2(weight_decay))(input_tensor)
        self.i = self.i+1

        tmp = keras.layers.Conv2D(128, kernel_size=(
            3, 3), padding='same', name='cv_2d_bot_%d' % self.i, kernel_regularizer=l2(weight_decay))(tmp)
        self.i = self.i+1

        tmp = keras.layers.Conv2D(256, kernel_size=(
            1, 1), padding='same', name='cv_2d_bot_%d' % self.i, kernel_regularizer=l2(weight_decay))(tmp)
        self.i = self.i+1

        input_tensor = keras.layers.Conv2D(256, kernel_size=(
            1, 1), padding='same', name='cv_2d_bot_%d' % self.i, kernel_regularizer=l2(weight_decay))(input_tensor)
        
        tmp = keras.layers.Add(name='cv_add_bot_%d' %
                               self.i)([tmp, input_tensor])
        tmp = keras.layers.Activation(
            'relu', name='cv_ac_bot_%d' % self.i)(tmp)
        self.i = self.i+1
        return tmp

    def get_model(self, weight_decay):
        pass
        base_model = keras.applications.densenet.DenseNet201(weights='imagenet', include_top=False,
                                                             input_shape=(512, 512, 3))
        C2 = base_model.layers[51].output
        #128 * 128 * 128
        C3 = base_model.layers[139].output
        #64 * 64 * 256
        C4 = base_model.layers[479].output
        #32 * 32 * 896
        C5 = base_model.output
        #16 * 16 * 1920

        P5 = keras.layers.Conv2D(
            256, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay))(C5)
        #P5 = keras.layers.BatchNormalization()(P5)
        P5 = keras.layers.Activation('relu')(P5)
        # --------add to c4
        upC5 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(P5)
        upC5 = keras.layers.Conv2D(
            256, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay))(upC5)
        #upC5 = keras.layers.BatchNormalization()(upC5)
        upC5 = keras.layers.Activation('relu')(upC5)
        # upC5 shape 32*32*256
        C4 = keras.layers.Conv2D(
            256, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay))(C4)
        #C4 = keras.layers.BatchNormalization()(C4)
        C4 = keras.layers.Activation('relu')(C4)
        P4 = keras.layers.Add()([upC5, C4])

        # ----------add to c3
        upC4 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(P4)
        upC4 = keras.layers.Conv2D(
            256, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay))(upC4)
        #upC4 = keras.layers.BatchNormalization()(upC4)
        upC4 = keras.layers.Activation('relu')(upC4)
        # upC4 shape 64*64*256
        C3 = keras.layers.Conv2D(
            256, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay))(C3)
        #C3 = keras.layers.BatchNormalization()(C3)
        C3 = keras.layers.Activation('relu')(C3)
        P3 = keras.layers.Add()([upC4, C3])

        # ----------add to c2
        upC3 = keras.layers.UpSampling2D(size=(2, 2), data_format=None)(P3)
        upC3 = keras.layers.Conv2D(
            256, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay))(upC3)
        #upC3 = keras.layers.BatchNormalization()(upC3)
        upC3 = keras.layers.Activation('relu')(upC3)
        # upC3 shape 128*128*256
        C2 = keras.layers.Conv2D(
            256, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay))(C2)
        #C2 = keras.layers.BatchNormalization()(C2)
        C2 = keras.layers.Activation('relu')(C2)
        P2 = keras.layers.Add()([upC3, C2])

        # output heatmap
        out_P2 = keras.layers.Conv2D(
            256, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay))(P2)
        #out_P2 = keras.layers.BatchNormalization()(out_P2)
        out_P2 = keras.layers.Activation('relu')(out_P2)

        out_P2 = keras.layers.Conv2D(
            self.point_num, kernel_size=(3, 3), padding='same', activation='sigmoid',
            name='pred_p2', kernel_regularizer=l2(weight_decay))(out_P2)

        out_P3 = keras.layers.Conv2D(
            256, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay))(P3)
        #out_P3 = keras.layers.BatchNormalization()(out_P3)
        out_P3 = keras.layers.Activation('relu')(out_P3)

        out_P3 = keras.layers.Conv2D(
            self.point_num, kernel_size=(3, 3), padding='same', activation='sigmoid',
            name='pred_p3', kernel_regularizer=l2(weight_decay))(out_P3)

        out_P4 = keras.layers.Conv2D(
            256, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay))(P4)
        #out_P4 = keras.layers.BatchNormalization()(out_P4)
        out_P4 = keras.layers.Activation('relu')(out_P4)

        out_P4 = keras.layers.Conv2D(
            self.point_num, kernel_size=(3, 3), padding='same', activation='sigmoid',
            name='pred_p4', kernel_regularizer=l2(weight_decay))(out_P4)

        out_P5 = keras.layers.Conv2D(
            256, kernel_size=(1, 1), kernel_regularizer=l2(weight_decay))(P5)
        #out_P5 = keras.layers.BatchNormalization()(out_P5)
        out_P5 = keras.layers.Activation('relu')(out_P5)

        out_P5 = keras.layers.Conv2D(
            self.point_num, kernel_size=(3, 3), padding='same', activation='sigmoid',
            name='pred_p5', kernel_regularizer=l2(weight_decay))(out_P5)

        P3 = self.res_bottleneck(P3, weight_decay)
        P3 = keras.layers.UpSampling2D(
            size=(2, 2), data_format=None, name='up_re_%d' % self.i)(P3)

        P4 = self.res_bottleneck(P4, weight_decay)
        P4 = self.res_bottleneck(P4, weight_decay)
        P4 = keras.layers.UpSampling2D(
            size=(4, 4), data_format=None, name='up_re_%d' % self.i)(P4)

        P5 = self.res_bottleneck(P5, weight_decay)
        P5 = self.res_bottleneck(P5, weight_decay)
        P5 = self.res_bottleneck(P5, weight_decay)
        P5 = keras.layers.UpSampling2D(
            size=(8, 8), data_format=None, name='up_re_%d' % self.i)(P5)

        output = keras.layers.Concatenate(axis=-1)([P2, P3, P4, P5])
        output = self.res_bottleneck(output, weight_decay)
        output = keras.layers.Conv2D(self.point_num, kernel_size=(
            3, 3), padding='same', activation='sigmoid', name='final_output', kernel_regularizer=l2(weight_decay))(output)

        model = Model(inputs=base_model.input, outputs=[
                      output, out_P2, out_P3, out_P4, out_P5])
        return model

    def train(self, train_gen, val_gen, save_model_path, data_len, FREEZ_BASE_MODEL=False, BATCH_SIZE=32, VAL_NUM=800, gpu_num=1, model_name=None, epochs=20, lr_rate=1e-3):
        if FREEZ_BASE_MODEL:
            # 479 densnet 745 globalnet
            for layer in self.model.layers[:745]:
                layer.trainable = False
            for layer in self.model.layers[745:]:
                layer.trainable = True
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_final_output_loss', patience=4)
        csv_logger = keras.callbacks.CSVLogger(
            save_model_path+'logger.csv', append=True)
        if self.gpu_num == 1:
            cbk = keras.callbacks.ModelCheckpoint(
                save_model_path+'.model', verbose=0, save_best_only=True, monitor = 'val_final_output_loss')
            tf_cbk = keras.callbacks.TensorBoard(log_dir='logs/'+save_model_path+'/', batch_size=BATCH_SIZE)
            self.model.compile(optimizer=keras.optimizers.SGD(lr=lr_rate, momentum=0.9),
                               loss={'final_output': custom_loss, 'pred_p2': custom_loss, 'pred_p3': custom_loss,
                                     'pred_p4': custom_loss, 'pred_p5': custom_loss},
                               loss_weights=[5, 3, 2, 1, 1])
            self.model.fit_generator(train_gen, validation_data=val_gen,
                                     steps_per_epoch=(
                                         data_len-VAL_NUM)/BATCH_SIZE,
                                     validation_steps=VAL_NUM/BATCH_SIZE, epochs=epochs/2,
                                     callbacks=[early_stopping, csv_logger, cbk,tf_cbk], use_multiprocessing=False)

            self.model.compile(optimizer=keras.optimizers.SGD(lr=lr_rate/10, momentum=0.9),
                               loss={'final_output': custom_loss, 'pred_p2': custom_loss, 'pred_p3': custom_loss,
                                     'pred_p4': custom_loss, 'pred_p5': custom_loss},
                               loss_weights=[5, 0.1, 0.1, 0.1, 0.1])
            self.model.fit_generator(train_gen, validation_data=val_gen,
                                     steps_per_epoch=(
                                         data_len-VAL_NUM)/BATCH_SIZE,
                                     validation_steps=VAL_NUM/BATCH_SIZE, epochs=epochs,
                                     callbacks=[early_stopping, csv_logger, cbk,tf_cbk],
                                     use_multiprocessing=False,initial_epoch = epochs/2)