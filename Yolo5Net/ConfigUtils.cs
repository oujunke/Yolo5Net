using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.NumPy;

namespace Yolo5Net
{
    public class ConfigUtils
    {
        public static float[] Width = new[] { 0.50f, 0.75f, 1.0f, 1.25f };
        public static float[] Depth = new[] { 0.33f, 0.67f, 1.0f, 1.33f };

        public static List<string> Versions = new List<string> { "s", "m", "l", "x" };
        public static string DataDir = Path.Combine("..", "Dataset", "COCO");

        public static float Threshold = 0.3f;
        public static int MaxBoxes = 150;
        public static string ImageDir = "images";
        public static string LabelDir = "labels";

        public static int NumEpochs = 300;
        public static int BatchSize = 32;
        public static int ImageSize = 640;
        public static Dictionary<string, int> ClassDict = new Dictionary<string, int> { { "person", 0 }, { "bicycle", 1 }, { "car", 2 }, { "motorcycle", 3 }, { "airplane", 4 }, { "bus", 5 }, { "train", 6 }, { "truck", 7 }, { "boat", 8 }, { "traffic light", 9 }, { "fire hydrant", 10 }, { "stop sign", 11 }, { "parking meter", 12 }, { "bench", 13 }, { "bird", 14 }, { "cat", 15 }, { "dog", 16 }, { "horse", 17 }, { "sheep", 18 }, { "cow", 19 }, { "elephant", 20 }, { "bear", 21 }, { "zebra", 22 }, { "giraffe", 23 }, { "backpack", 24 }, { "umbrella", 25 }, { "handbag", 26 }, { "tie", 27 }, { "suitcase", 28 }, { "frisbee", 29 }, { "skis", 30 }, { "snowboard", 31 }, { "sports ball", 32 }, { "kite", 33 }, { "baseball bat", 34 }, { "baseball glove", 35 }, { "skateboard", 36 }, { "surfboard", 37 }, { "tennis racket", 38 }, { "bottle", 39 }, { "wine glass", 40 }, { "cup", 41 }, { "fork", 42 }, { "knife", 43 }, { "spoon", 44 }, { "bowl", 45 }, { "banana", 46 }, { "apple", 47 }, { "sandwich", 48 }, { "orange", 49 }, { "broccoli", 50 }, { "carrot", 51 }, { "hot dog", 52 }, { "pizza", 53 }, { "donut", 54 }, { "cake", 55 }, { "chair", 56 }, { "couch", 57 }, { "potted plant", 58 }, { "bed", 59 }, { "dining table", 60 }, { "toilet", 61 }, { "tv", 62 }, { "laptop", 63 }, { "mouse", 64 }, { "remote", 65 }, { "keyboard", 66 }, { "cell phone", 67 }, { "microwave", 68 }, { "oven", 69 }, { "toaster", 70 }, { "sink", 71 }, { "refrigerator", 72 }, { "book", 73 }, { "clock", 74 }, { "vase", 75 }, { "scissors", 76 }, { "teddy bear", 77 }, { "hair drier", 78 }, { "toothbrush", 79 } };

        public static string Version = "s";
        public static NDArray Anchors = new NDArray(new float[,] { { 8, 9 }, { 16, 24 },   {28, 58 },
                       {41, 25 },  {58, 125 },  {71, 52 },
                       { 129, 97}, {163, 218}, {384, 347} });
    }
}
