import os, cv2
import numpy as np
from network import *
from network_configure import conf_unet
from utils.predict_utils import *
import time

class UNet2D(object):

    def __init__(self, base_dir, name, in_dim = 1, out_dim = 1,
                 train_config = {'base_learning_rate': 0.0004,
                                 'epoch': 10,
                                 'batch_size': 8,
                                 'step_per_epoch': None,
                                 'epoch_per_val': 2,
                                 'weight_decay': 0.0,
                                 'probalistic': False,
                                 'loss': 'mae',
                                 'lr_decay':{'decay_steps':1e4, 
                                             'decay_rate':0.5, 
                                             'staircase':True}},
                                 **kwargs):

        self.base_dir = base_dir
        self.name = name
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.base_lr = train_config['base_learning_rate']
        self.epoch = train_config['epoch']
        self.batch_size = train_config['batch_size']
        self.step_per_epoch = train_config['step_per_epoch']
        self.weight_decay = train_config['weight_decay']
        self.lr_decay = train_config['lr_decay']
        self.train_config = train_config
        self.net = UNet(conf_unet)

    def _model_fn(self, features, labels, mode):
        out, att, q, k = self.net(features, mode == tf.estimator.ModeKeys.TRAIN)
        out = batch_norm(out, mode == tf.estimator.ModeKeys.TRAIN)
        out = relu(out)
        preds = convolution_2D(out, self.out_dim, 1, 1, False, name = 'out_conv')

        if mode == tf.estimator.ModeKeys.PREDICT:
            if self.train_config['loss'] not in ['mae', 'mse']:
                preds = tf.argmax(preds, axis=-1)
            pred_lst = {'in': features, 'preds':preds, 'att':att, 'q':q, 'k':k}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_lst)

        if self.train_config['loss'] in ['mae', 'mse']:
            loss_mae = tf.identity(tf.reduce_mean(tf.abs(preds-labels), axis=None), name='mae_loss')
            loss_mse = tf.identity(tf.reduce_mean(tf.square(preds-labels), axis=None), name='mse_loss')
            tf.summary.scalar('mae_loss', loss_mae)
            tf.summary.scalar('mse_loss', loss_mse)
        else:
            loss_clf = tf.losses.sparse_softmax_cross_entropy(labels, preds)

        if self.train_config['probalistic']:
            sigma = convolution_2D(out, self.out_dim, 1, 1, False, name = 'out_sigma_conv')
            sigma = tf.nn.softplus(sigma) + 1e-3
            loss = tf.reduce_mean(tf.truediv(tf.abs(preds-labels), sigma) + 
                                        tf.log(sigma)) + self.weight_decay * tf.add_n([tf.nn.l2_loss(v) 
                                        for v in tf.trainable_variables() if 'kernel' in v.name])

        else:
            loss = (loss_mae if self.train_config['loss']=='mae'
                    else loss_mse if self.train_config['loss']=='mse'
                    else loss_clf
                   )
            loss = loss + self.weight_decay * tf.add_n([tf.nn.l2_loss(v) 
                                        for v in tf.trainable_variables() if 'kernel' in v.name])

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.base_lr, global_step, 
                            self.lr_decay['decay_steps'], self.lr_decay['decay_rate'], self.lr_decay['staircase'])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step)
        else:
            train_op = None
        
        metrics = {'mae':tf.metrics.mean_absolute_error(labels, preds), 
                   'mse':tf.metrics.mean_squared_error(labels, preds)}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=preds, loss=loss, train_op=train_op, 
                                          eval_metric_ops=metrics)
    
    def _model_fn_clf(self, features, labels, mode):
        out, att, q, k = self.net(features, mode == tf.estimator.ModeKeys.TRAIN)
        out = batch_norm(out, mode == tf.estimator.ModeKeys.TRAIN)
        out = relu(out)
        preds = convolution_2D(out, 256, 1, 1, False, name = 'out_conv')

        if mode == tf.estimator.ModeKeys.PREDICT:
            preds = tf.argmax(preds, axis=-1)
            pred_lst = {'in': features, 'preds':preds, 'att':att, 'q':q, 'k':k}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_lst)

        labels = tf.squeeze(tf.to_int32(tf.round(labels *255.0)))
        loss_clf = tf.losses.sparse_softmax_cross_entropy(labels, preds)
        loss_clf /= -1.0 * math.log(1.0 / 256.0)
        loss = loss_clf + self.weight_decay * tf.add_n([tf.nn.l2_loss(v) 
                                    for v in tf.trainable_variables() if 'kernel' in v.name])

        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step = tf.train.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(self.base_lr, global_step, 
                            self.lr_decay['decay_steps'], self.lr_decay['decay_rate'], self.lr_decay['staircase'])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step)
        else:
            train_op = None
        
        return tf.estimator.EstimatorSpec(mode=mode, predictions=preds, loss=loss, train_op=train_op)
    
    
    def _model_fn_visual(self, features, labels, mode):
        out, att, q, k = self.net(features, mode == tf.estimator.ModeKeys.TRAIN)
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'att': att, 'q':q, 'k':k})
    

    def _input_fn(self, sources, targets, patch_size, batch_size, shuffle=True):

        def generator():
            while(True):
                idx = np.random.randint(len(sources))
                source, target = sources[idx], targets[idx]

                valid_shape = source.shape[:-1] - np.array(patch_size)
                y = np.random.randint(0, valid_shape[0])
                x = np.random.randint(0, valid_shape[1])
                s = (slice(y, y+patch_size[0]), 
                     slice(x, x+patch_size[1]))
                source_patch = source[s]
                target_patch = target[s]
                yield source_patch, target_patch

        output_types = (tf.float32, tf.float32)
        output_shapes = (tf.TensorShape([s for s in patch_size] + [self.in_dim]), 
                         tf.TensorShape(patch_size + [self.out_dim]))
        dataset = tf.data.Dataset.from_generator(generator, 
                        output_types=output_types, output_shapes=output_shapes)
        dataset = dataset.batch(batch_size).prefetch(1)
        return dataset


    def train(self, source_lst, target_lst, patch_size, validation=None, save_steps=1000, log_steps=200, steps=50000, batch_size=32, seed=0):
        
        tf.set_random_seed(seed)
        np.random.seed(seed)

        ses_config = tf.ConfigProto()
        ses_config.gpu_options.allow_growth = True

        run_config = tf.estimator.RunConfig(model_dir=self.base_dir+'/'+self.name, 
                                            save_checkpoints_steps=save_steps,
                                            session_config=ses_config, 
                                            keep_checkpoint_max = 20,
                                            log_step_count_steps=log_steps,
                                            save_summary_steps=log_steps)

        transformer = tf.estimator.Estimator(model_fn=self._model_fn, 
                                            model_dir=self.base_dir+'/'+self.name, config=run_config)
#         logging = tf.train.LoggingTensorHook(tensors={'mae_loss':'mae_loss', 'mse_loss':'mse_loss'}, 
#                                              every_n_iter=log_steps)
        input_fn = lambda: self._input_fn(source_lst, target_lst, patch_size, batch_size=batch_size)
        
        if validation:
            train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=steps)#, hooks=[logging])
            val_input_fn = tf.estimator.inputs.numpy_input_fn(x=validation[0], y=validation[1], batch_size=1, 
                                                              num_epochs=1, shuffle=False)
            eval_spec = tf.estimator.EvalSpec(input_fn=val_input_fn, throttle_secs=60)#, exporters=exporter)
            tf.estimator.train_and_evaluate(transformer, train_spec, eval_spec)
        else:
            transformer.train(input_fn=input_fn, steps=steps)#, hooks=[logging])
            

    def predict(self, image, resizer=PadAndCropResizer(), checkpoint_path=None):

        tf.logging.set_verbosity(tf.logging.ERROR)
        transformer = tf.estimator.Estimator(model_fn=self._model_fn, 
                                            model_dir=self.base_dir+'/'+self.name)

        image = resizer.before(image, 2 ** (self.net.depth+1), exclude=None)
        input_fn = tf.estimator.inputs.numpy_input_fn(x=image[None,...,None], batch_size=1, num_epochs=1, shuffle=False)
        image = list(transformer.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0]
        image = image['preds'][...,0]
        image = resizer.after(image, exclude=None)
            
        return image
    
    def visual_attention(self, image, resizer=PadAndCropResizer(), checkpoint_path=None):

        tf.logging.set_verbosity(tf.logging.ERROR)
        transformer = tf.estimator.Estimator(model_fn=self._model_fn_visual, 
                                            model_dir=self.base_dir+'/'+self.name)

        image = resizer.before(image, 2 ** (self.net.depth+1), exclude=0)
        input_fn = tf.estimator.inputs.numpy_input_fn(x=image[...,None], batch_size=1, num_epochs=1, shuffle=False)
        outs = list(transformer.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))
            
        return outs
    
    def get_att(self, image, checkpoint_path):
        tramsformer = tf.estimator.Estimator(model_fn=self._model_fn, 
                                             model_dir=self.base_dir+'/'+self.name)
        pred = list(tramsformer.predict(input_fn=self._input_fn(image, None, 'PRED', 1), 
                                        checkpoint_path=checkpoint_path))
        return pred
    
    def crop_predict(self, image, size, margin, resizer=PadAndCropResizer(), checkpoint_path=None):

        tf.logging.set_verbosity(tf.logging.ERROR)
        transformer = tf.estimator.Estimator(model_fn=self._model_fn, 
                                            model_dir=self.base_dir+'/'+self.name)
        
        out_image = np.empty(image.shape, dtype='float32')
        preds = []
        for src_s, trg_s, mrg_s in get_coord(image.shape, size, margin):
            patch = resizer.before(image[src_s], 2 ** (self.net.depth), exclude=None)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=patch[None, ..., None], batch_size=1, num_epochs=1, shuffle=False)
            pred = list(transformer.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0]
            preds.append(pred)
            patch = pred['preds'][...,0]
            patch = resizer.after(patch, exclude=None)
            out_image[trg_s] = patch[mrg_s]
        return out_image, preds
    
    def crop_equav(self, image, size, margin, sp_trans, resizer=PadAndCropResizer(), checkpoint_path=None):

        tf.logging.set_verbosity(tf.logging.ERROR)
        transformer = tf.estimator.Estimator(model_fn=self._model_fn, 
                                            model_dir=self.base_dir+'/'+self.name)
        
        out_image = np.empty(image.shape, dtype='float32')
        preds = []
        for src_s, trg_s, mrg_s in get_coord(image.shape, size, margin):
            patch = resizer.before(image[src_s], 2 ** (self.net.depth), exclude=None)
            patch = sp_trans.before(patch)
            input_fn = tf.estimator.inputs.numpy_input_fn(x=patch[None, ..., None], batch_size=1, num_epochs=1, shuffle=False)
            pred = list(transformer.predict(input_fn=input_fn, checkpoint_path=checkpoint_path))[0]
            preds.append(pred)
            patch = pred['preds'][...,0]
            patch = sp_trans.after(patch)
            patch = resizer.after(patch, exclude=None)
            out_image[trg_s] = patch[mrg_s]
        return out_image, preds
    
    
def get_coord(shape, size, margin):
    n_tiles_i = int(np.ceil((shape[1]-size)/float(size-2*margin)))
    n_tiles_j = int(np.ceil((shape[0]-size)/float(size-2*margin)))
    for i in range(n_tiles_i+1):
        src_start_i = i*(size-2*margin) if i<n_tiles_i else (shape[1]-size)
        src_end_i = src_start_i+size
        left_i = margin if i>0 else 0
        right_i = margin if i<n_tiles_i else 0
        for j in range(n_tiles_j+1):
            src_start_j = j*(size-2*margin) if j<n_tiles_j else (shape[0]-size)
            src_end_j = src_start_j+size
            left_j = margin if j>0 else 0
            right_j = margin if j<n_tiles_j else 0
            src_s = (slice(src_start_j, src_end_j), slice(src_start_i, src_end_i))
            trg_s = (slice(src_start_j+left_j, src_end_j-right_j), slice(src_start_i+left_i, src_end_i-right_i))
            mrg_s = (slice(left_j, -right_j if right_j else None), slice(left_i, -right_i if right_i else None))
            yield src_s, trg_s, mrg_s