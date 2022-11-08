using Tensorflow;
using static Tensorflow.Binding;
namespace Yolo5Net
{
    public class Yolo5
    {
        public static void Run()
        {
            /*var p = "D:\\GitCode\\YOLOv5-tf\\pascalvoc\\VOCdevkit\\VOC2012\\JPEGImages";
            var fi = new DirectoryInfo(p).GetFiles().ToList();
            var index =(int)( fi.Count * 0.8);
            var trl= fi.Take(index).Select(f=>f.Name.Replace(f.Extension,"")).ToList();
            File.WriteAllLines("D:\\GitCode\\YOLOv5-tf\\pascalvoc\\VOCdevkit\\VOC2012\\train.txt", trl);
            File.WriteAllLines("D:\\GitCode\\YOLOv5-tf\\pascalvoc\\VOCdevkit\\VOC2012\\val.txt", fi.Skip(index).Select(f => f.Name.Replace(f.Extension, "")).ToList());*/
            var model=NetUtils.BuildModel();
            Graph graph = Binding.tf.Graph().as_default();
            model.summary();
            model.save("modelsYolo");
        }
        public static void Train()
        {
            //var strategy = tf.distribute.MirroredStrategy();
        }
    }
}