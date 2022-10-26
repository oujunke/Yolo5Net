using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.Operations;
using Tensorflow.Operations.Initializers;
using static HDF.PInvoke.H5Z;

namespace Yolo5Net
{
    internal class NetUtils
    {
        private static IInitializer initializer = Binding.tf.random_normal_initializer(0f, 0.01f);
        private static IRegularizer l2 = KerasApi.keras.regularizers.l2(0.00004f);
        public static Tensor Conv(Tensor x, int filters, int k = 1, int s = 1)
        {
            string padding;
            if (s == 2)
            {
                x = (Tensor)KerasApi.keras.layers.ZeroPadding2D(new int[2, 2]
                {
                    { 1, 0 },
                    { 1, 0 }
                }).Apply(x);
                padding = "valid";
            }
            else
            {
                padding = "same";
            }
            Tensors tensors = KerasApi.keras.layers.Conv2D(filters, k, s, padding, use_bias: false, kernel_regularizer: l2, kernel_initializer: initializer).Apply(x);
            tensors = KerasApi.keras.layers.BatchNormalization(momentum: 0.03f).Apply(tensors);
            tensors = KerasApi.keras.layers.Swish().Apply(tensors);
            return (Tensor)tensors;
        }
        public static Tensor Residual(Tensor x, int filters, bool add = true)
        {
            var inputs = x;
            if (add)
            {
                x = Conv(x, filters, 1);
                x = Conv(x, filters, 3);
                x = inputs + x;
            }
            else
            {
                x = Conv(x, filters, 1);
                x = Conv(x, filters, 3);
            }
            return x;
        }
        public static Tensor Csp(Tensor x, int filters, int n, bool add = true)
        {
            var y = Conv(x, filters / 2);
            for (int i = 0; i < n; i++)
            {
                y = Residual(y, filters / 2, add);
            }
            x = Conv(x, filters / 2);
            x = KerasApi.keras.layers.Concatenate().Apply(new Tensors(x, y));
            x = Conv(x, filters);
            return x;
        }

        public static Functional BuildModel(bool training = true)
        {
            var vindex = ConfigUtils.Versions.IndexOf(ConfigUtils.Version);
            var depth = ConfigUtils.Depth[vindex];
            var width = ConfigUtils.Width[vindex];

            var inputs = KerasApi.keras.layers.Input(new Shape(ConfigUtils.ImageSize, ConfigUtils.ImageSize, 3));
            var x = gen_ops.space_to_batch(inputs, new Tensor(new int[2, 2]), 2);
            x = Conv(x, (int)Math.Round(width * 64), 3);
            x = Conv(x, (int)Math.Round(width * 128), 3, 2);
            x = Csp(x, (int)Math.Round(width * 128), (int)Math.Round(depth * 3));

            x = Conv(x, (int)Math.Round(width * 256), 3, 2);
            x = Csp(x, (int)Math.Round(width * 256), (int)Math.Round(depth * 9));
            var x1 = x;

            x = Conv(x, (int)Math.Round(width * 512), 3, 2);
            x = Csp(x, (int)Math.Round(width * 512), (int)Math.Round(depth * 9));
            var x2 = x;

            x = Conv(x, (int)Math.Round(width * 1024), 3, 2);
            x = Conv(x, (int)Math.Round(width * 512), 1, 1);

            x = KerasApi.keras.layers.Concatenate().Apply(new[] { x, nn_ops.max_pool(x, new[] { 1, 5, 5, 1 }, new[] { 1, 1, 1, 1 }, "SAME"), nn_ops.max_pool(x, new[] { 1, 9, 9, 1 }, new[] { 1, 1, 1, 1 }, "SAME"), nn_ops.max_pool(x, new[] { 1, 13, 13, 1 }, new[] { 1, 1, 1, 1 }, "SAME") });
            x = Conv(x, (int)Math.Round(width * 1024), 1, 1);
            x = Csp(x, (int)Math.Round(width * 1024), (int)Math.Round(depth * 3), false);

            x = Conv(x, (int)Math.Round(width * 512), 1);
            var x3 = x;
            x = KerasApi.keras.layers.UpSampling2D().Apply(x);
            x = KerasApi.keras.layers.Concatenate().Apply(new[] { x, x2 });
            x = Csp(x, (int)Math.Round(width * 512), (int)Math.Round(depth * 3), false);

            x = Conv(x, (int)Math.Round(width * 256), 1);
            var x4 = x;
            x = KerasApi.keras.layers.UpSampling2D().Apply(x);
            x = KerasApi.keras.layers.Concatenate().Apply(new[] { x, x1 });
            x = Csp(x, (int)Math.Round(width * 256), (int)Math.Round(depth * 3), false);
            var p3 = KerasApi.keras.layers.Conv2D(3 * (ConfigUtils.ClassDict.Count + 5), 1,
                                kernel_initializer: initializer, kernel_regularizer: l2).SetName($"'p3_{ConfigUtils.ClassDict.Count}'").Apply(x);

            x = Conv(x, (int)Math.Round(width * 256), 3, 2);
            x = KerasApi.keras.layers.Concatenate().Apply(new[] { x, x4 });
            x = Csp(x, (int)Math.Round(width * 512), (int)Math.Round(depth * 3), false);
            var p4 = KerasApi.keras.layers.Conv2D(3 * (ConfigUtils.ClassDict.Count + 5), 1,
                               kernel_initializer: initializer, kernel_regularizer: l2).SetName($"'p4_{ConfigUtils.ClassDict.Count}'").Apply(x);

            x = Conv(x, (int)Math.Round(width * 512), 3, 2);
            x = KerasApi.keras.layers.Concatenate().Apply(new[] { x, x3 });
            x = Csp(x, (int)Math.Round(width * 1024), (int)Math.Round(depth * 3), false);
            var p5 = KerasApi.keras.layers.Conv2D(3 * (ConfigUtils.ClassDict.Count + 5), 1,
                                 kernel_initializer: initializer, kernel_regularizer: l2).SetName($"'p4_{ConfigUtils.ClassDict.Count}'").Apply(x);
            if (training) {
                return KerasApi.keras.Model(inputs, new Tensors( p5, p4, p3 )); 
            }
            else {
                return KerasApi.keras.Model(inputs, Predict()([p5, p4, p3]))}
        }
        public static void ProcessLayer(Tensors feature_map, Tensor tensor)
        {
            var grid_size = feature_map.shape[1..3];
            nn_ops
    ratio = tf.cast(tf.constant([config.image_size, config.image_size]) / grid_size, tf.float32)
    rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

    feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + len(config.class_dict)])

    box_centers, box_sizes, conf, prob = tf.split(feature_map, [2, 2, 1, len(config.class_dict)], axis = -1)
    box_centers = tf.nn.sigmoid(box_centers)

    grid_x = tf.range(grid_size[1], dtype = tf.int32)
    grid_y = tf.range(grid_size[0], dtype = tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(grid_x, (-1, 1))
    y_offset = tf.reshape(grid_y, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis = -1)
                x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * ratio[::- 1]

    box_sizes = tf.exp(box_sizes) * rescaled_anchors
    box_sizes = box_sizes * ratio[::- 1]

    boxes = tf.concat([box_centers, box_sizes], axis = -1)

    return x_y_offset, boxes, conf, prob
        }
        public static Functional Predict(params Tensors[] inputs)
        {

            for (int i = 0; i < inputs.Length; i++)
            {
                var feature_map = inputs[0];
                var anchors = ConfigUtils.Anchors.slice(new Slice((3-i-1)*3, (3 - i) * 3));
                x_y_offset, box, conf, prob = result
            grid_size = tf.shape(x_y_offset)[:2]
            box = tf.reshape(box, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf = tf.reshape(conf, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob = tf.reshape(prob, [-1, grid_size[0] * grid_size[1] * 3, len(config.class_dict)])
            boxes_list.append(box)
            conf_list.append(tf.sigmoid(conf))
            prob_list.append(tf.sigmoid(prob))
            }
        boxes = tf.concat(boxes_list, axis = 1)
        conf = tf.concat(conf_list, axis = 1)
        prob = tf.concat(prob_list, axis = 1)

        center_x, center_y, w, h = tf.split(boxes, [1, 1, 1, 1], axis = -1)
        x_min = center_x - w / 2
        y_min = center_y - h / 2
        x_max = center_x + w / 2
        y_max = center_y + h / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis = -1)

        outputs = tf.map_fn(fn = compute_nms,
                            elems =[boxes, conf * prob],
                            dtype =['float32', 'float32', 'int32'],
                            parallel_iterations = 100)

        return outputs
        }
    }
}
