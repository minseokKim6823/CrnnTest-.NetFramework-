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

        public OcrResult(string serial, string amount, string micr)
        {
            SerialText = serial;
            AmountText = amount;
            MicrText = micr;
        }

        public override string ToString()
        {
            return $"Serial1: {SerialText}\nAmoun1t: {AmountText}\nMICR1: {MicrText}";
        }
    }
}
