from gunpowder.zoo.tensorflow import unet, conv_pass
import tensorflow as tf
import json

def create_network(input_shape, name):

    tf.reset_default_graph()

    # create a placeholder for the 3D raw input tensor
    raw = tf.placeholder(tf.float32, shape=input_shape)

    # create a U-Net
    raw_batched = tf.reshape(raw, (1, 1) + input_shape)
    unet_output = unet(raw_batched, 6, 4, [[1,3,3],[1,3,3],[1,3,3]])

    # add a convolution layer to create 3 output maps representing affinities
    # in z, y, and x
    pred_affs_batched = conv_pass(
        unet_output,
        kernel_size=1,
        num_fmaps=3,
        num_repetitions=1,
        activation='sigmoid')

    # get the shape of the output
    output_shape_batched = pred_affs_batched.get_shape().as_list()
    output_shape = output_shape_batched[1:] # strip the batch dimension

    # the 4D output tensor (3, depth, height, width)
    pred_affs = tf.reshape(pred_affs_batched, output_shape)

    # create a placeholder for the corresponding ground-truth affinities
    gt_affs = tf.placeholder(tf.float32, shape=output_shape)

    # create a placeholder for per-voxel loss weights
    loss_weights = tf.placeholder(
        tf.float32,
        shape=output_shape)

    # compute the loss as the weighted mean squared error between the
    # predicted and the ground-truth affinities
    loss = tf.losses.mean_squared_error(
        gt_affs,
        pred_affs,
        loss_weights)

    # use the Adam optimizer to minimize the loss
    opt = tf.train.AdamOptimizer(
        learning_rate=0.5e-4,
        beta1=0.95,
        beta2=0.999,
        epsilon=1e-8)
    optimizer = opt.minimize(loss)

    # store the network in a meta-graph file
    tf.train.export_meta_graph(filename=name + '.meta')

    # store network configuration for use in train and predict scripts
    config = {
        'raw': raw.name,
        'pred_affs': pred_affs.name,
        'gt_affs': gt_affs.name,
        'loss_weights': loss_weights.name,
        'loss': loss.name,
        'optimizer': optimizer.name,
        'input_shape': input_shape,
        'output_shape': output_shape[1:]
    }
    with open(name + '_config.json', 'w') as f:
        json.dump(config, f)

if __name__ == "__main__":

    # create a network for training
    create_network((84, 268, 268), 'train_net')

    # create a larger network for faster prediction
    create_network((120, 322, 322), 'test_net')
