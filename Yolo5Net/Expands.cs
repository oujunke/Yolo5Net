using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow.Keras.Engine;

namespace Yolo5Net
{
    public static class Expands
    {
        public static Layer SetName(this Layer layer,string name)
        {
            layer.SetName(name);
            return layer; 
        }
    }
}
