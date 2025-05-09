using System;
using System.Windows.Forms;
using Emgu.CV;

namespace CrnnTest
{
    public class OcrResult
    {
        public string SerialText { get; set; }
        public string AmountText { get; set; }
        public string MicrText { get; set; }
        public Mat BoxImg { get; set; }

        public OcrResult(string serial, string amount, string micr, Mat boxImg)
        {
            SerialText = serial;
            AmountText = amount;
            MicrText = micr;
            BoxImg = boxImg;
        }

        public override string ToString()
        {
            return $"Serial: {SerialText} \r\nAmount: {AmountText} \r\nMICR: {MicrText}";
        }
    }
}
