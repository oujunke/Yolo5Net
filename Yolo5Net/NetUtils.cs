using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Engine;
using Tensorflow.NumPy;
using Tensorflow.Operations;
using Tensorflow.Operations.Initializers;
using static HDF.PInvoke.H5Z;
using static System.Formats.Asn1.AsnWriter;
using static Tensorflow.ApiDef.Types;
using static Tensorflow.Binding;
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
            tf.init_scope();
            for (int i = 0; i < 1000; i++)
            {
                //Thread.Sleep(1000);
            }
            var vindex = ConfigUtils.Versions.IndexOf(ConfigUtils.Version);
            var depth = ConfigUtils.Depth[vindex];
            var width = ConfigUtils.Width[vindex];

            var inputs = KerasApi.keras.layers.Input(new Shape(ConfigUtils.ImageSize, ConfigUtils.ImageSize, 3));
            Console.WriteLine(tf.VERSION);
            Tensor x = KerasApi.keras.layers.Conv2D(12, new Shape(2,2), new Shape(2,2), "same", use_bias: false, kernel_regularizer: l2, kernel_initializer: initializer).Apply(inputs);
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
                                kernel_initializer: initializer, kernel_regularizer: l2).Apply(x);

            x = Conv(x, (int)Math.Round(width * 256), 3, 2);
            x = KerasApi.keras.layers.Concatenate().Apply(new[] { x, x4 });
            x = Csp(x, (int)Math.Round(width * 512), (int)Math.Round(depth * 3), false);
            var p4 = KerasApi.keras.layers.Conv2D(3 * (ConfigUtils.ClassDict.Count + 5), 1,
                               kernel_initializer: initializer, kernel_regularizer: l2).Apply(x);

            x = Conv(x, (int)Math.Round(width * 512), 3, 2);
            x = KerasApi.keras.layers.Concatenate().Apply(new[] { x, x3 });
            x = Csp(x, (int)Math.Round(width * 1024), (int)Math.Round(depth * 3), false);
            var p5 = KerasApi.keras.layers.Conv2D(3 * (ConfigUtils.ClassDict.Count + 5), 1,
                                 kernel_initializer: initializer, kernel_regularizer: l2).Apply(x);
            if (training)
            {
                return KerasApi.keras.Model(inputs, new Tensors(p5, p4, p3));
            }
            else
            {
                return KerasApi.keras.Model(inputs, Predict(p5, p4, p3));
            }
        }
        public static (Tensor, Tensor, Tensor, Tensor) ProcessLayer(Tensors feature_map, Tensor anchors)
        {
            var grid_size = feature_map.shape[1..3];
            var ratio = math_ops.cast(constant_op.constant(new[] { ConfigUtils.ImageSize, ConfigUtils.ImageSize }) / grid_size, TF_DataType.TF_FLOAT);
            var rescaled_anchors = anchors.numpy().Select(na => new[] { na[0] / ratio[1], na[1] / ratio[0] }).ToArray();

            feature_map = array_ops.reshape(feature_map, new Tensor(new[] { -1, grid_size[0], grid_size[1], 3, 5 + ConfigUtils.ClassDict.Count }));
            var spRes = array_ops.split(feature_map, new Tensor(new[] { 2, 2, 1, ConfigUtils.ClassDict.Count }), -1);
            var box_centers = spRes[0];
            var box_sizes = spRes[1];
            var conf = spRes[2];
            var prob = spRes[3];
            box_centers = math_ops.sigmoid(box_centers);

            var grid_x = math_ops.range(grid_size[1], dtype: TF_DataType.TF_INT32);
            var grid_y = math_ops.range(grid_size[0], dtype: TF_DataType.TF_INT32);
            var mesRes = array_ops.meshgrid(new[] { grid_x, grid_y });
            grid_x = mesRes[0];
            grid_y = mesRes[1];
            var x_offset = array_ops.reshape(grid_x, new Tensor(new[] { -1, 1 }));
            var y_offset = array_ops.reshape(grid_y, new Tensor(new[] { -1, 1 }));
            var x_y_offset = array_ops.concat(new[] { x_offset, y_offset }, -1);
            x_y_offset = math_ops.cast(array_ops.reshape(x_y_offset, new Tensor(new[] { grid_size[0], grid_size[1], 1, 2 })), dtype: TF_DataType.TF_FLOAT);
            box_centers = box_centers + x_y_offset;
            var descRatio = gen_array_ops.reverse(ratio, -1);
            box_centers = box_centers * descRatio;


            box_sizes = gen_math_ops.exp(box_sizes) * rescaled_anchors;

            box_sizes = box_sizes * descRatio;


            var boxes = array_ops.concat(new[] { box_centers, box_sizes }, -1);

            return (x_y_offset, boxes, conf, prob);
        }
        public static Tensor Predict(params Tensors[] inputs)
        {
            List<Tensor> boxes_list = new List<Tensor>();
            List<Tensor> conf_list = new List<Tensor>();
            List<Tensor> prob_list = new List<Tensor>();
            for (int i = 0; i < inputs.Length; i++)
            {
                var feature_map = inputs[0];
                var anchors = ConfigUtils.Anchors.slice(new Slice((3 - i - 1) * 3, (3 - i) * 3));
                var (x_y_offset, box, conf2, prob2) = ProcessLayer(feature_map, anchors);
                var grid_size = x_y_offset.shape.Slice(0, 2);
                box = array_ops.reshape(box, new Tensor(new[] { -1, grid_size[0] * grid_size[1] * 3, 4 }));
                conf2 = array_ops.reshape(conf2, new Tensor(new[] { -1, grid_size[0] * grid_size[1] * 3, 1 }));
                prob2 = array_ops.reshape(prob2, new Tensor(new[] { -1, grid_size[0] * grid_size[1] * 3, ConfigUtils.ClassDict.Count }));
                boxes_list.append(box);
                conf_list.append(math_ops.sigmoid(conf2));
                prob_list.append(math_ops.sigmoid(prob2));
            }
            var boxes = array_ops.concat(boxes_list.ToArray(), 1);
            var conf = array_ops.concat(conf_list.ToArray(), 1);
            var prob = array_ops.concat(prob_list.ToArray(), 1);
            var spRes = array_ops.split(boxes, new Tensor(new[] { 1, 1, 1, 1 }), -1);
            var center_x = spRes[0];
            var center_y = spRes[1];
            var w = spRes[2];
            var h = spRes[3];
            var x_min = center_x - w / 2;
            var y_min = center_y - h / 2;
            var x_max = center_x + w / 2;
            var y_max = center_y + h / 2;

            boxes = array_ops.concat(new[] { x_min, y_min, x_max, y_max }, -1);

            var outputs = Operation.map_fn(ComputeNms,
                                 new Tensor(new[] { boxes, conf * prob }), TF_DataType.TF_INT32,
                                 100);

            return outputs;
        }
        public static Tensor NmsFn(Tensor boxes, Tensor score, Tensor label)
        {
            var score_indices = array_ops.where(gen_math_ops.greater(score, ConfigUtils.Threshold));
            var filtered_boxes = gen_ops.gather_nd(boxes, score_indices);
            var filtered_scores = gen_ops.gather(score, score_indices).slice(new Slice(null, null, 0));
            var nms_indices = gen_ops.non_max_suppression(filtered_boxes, filtered_scores, new Tensor(ConfigUtils.MaxBoxes), 0.1f);
            score_indices = gen_ops.gather(score_indices, nms_indices);
            label = gen_ops.gather_nd(label, score_indices);
            score_indices = array_ops.stack(new Tensor[] { score_indices.slice(new Slice(null, null, 0)), label }, 1);
            return score_indices;
        }
        public static Tensor ComputeNms(Tensor tensor)
        {
            Tensor boxes = tensor[0];
            Tensor classification = tensor[1];
            var count = classification.shape[1];
            List<Tensor> all_indices = new List<Tensor>();
            for (int i = 0; i < count; i++)
            {
                var scores = classification.slice(new Slice(0, 0, i));
                var labels = i * array_ops.ones((gen_ops.shape(scores)[0]), TF_DataType.TF_INT64);
                all_indices.append(NmsFn(boxes, scores, labels));
            }
            var indices = (Tensor)KerasApi.keras.layers.Concatenate(0).Apply(all_indices);
            var scores2 = gen_ops.gather_nd(classification, indices);
            var labels2 = indices.slice(new Slice(null, null));
            var (scores3, top_indices) = gen_ops.top_k(scores2, (int)gen_ops.minimum(new Tensor(ConfigUtils.MaxBoxes), new Tensor(scores2.shape[0])));

            indices = gen_ops.gather(indices.slice(new Slice(null, null, 0)), top_indices);
            boxes = gen_ops.gather(boxes, indices);
            labels2 = gen_ops.gather(labels2, top_indices);

            var pad_size = gen_ops.maximum(new Tensor(0), new Tensor(ConfigUtils.MaxBoxes - scores2.shape[0]));

            boxes = gen_ops.pad_v2(boxes, new Tensor(new int[,] { { 0, (int)pad_size }, { 0, 0 } }), new Tensor(-1));
            scores3 = gen_ops.pad_v2(scores3, new Tensor(new int[,] { { 0, (int)pad_size } }), new Tensor(-1));
            labels2 = gen_ops.pad_v2(labels2, new Tensor(new int[,] { { 0, (int)pad_size } }), new Tensor(-1));
            labels2 = gen_ops.cast(labels2, TF_DataType.TF_INT32);

            boxes.set_shape(new Tensor(new[] { ConfigUtils.MaxBoxes, 4 }));
            scores3.set_shape(new Tensor(new[] { ConfigUtils.MaxBoxes }));
            labels2.set_shape(new Tensor(new[] { ConfigUtils.MaxBoxes }));

            return new Tensor(new[] { boxes, scores3, labels2 });
        }
    }
}
