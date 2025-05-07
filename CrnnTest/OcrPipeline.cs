using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Structure;

namespace CrnnTest
{
    public class OcrPipeline
    {
        private OcrDetector detector;
        private OcrRecognizer rec1;
        private OcrRecognizer2 rec2;


        public OcrPipeline()
        {
            detector = new OcrDetector("en_PP-OCRv3_det_infer.onnx");
            rec1 = new OcrRecognizer("num_rec_250312.onnx");
            rec2 = new OcrRecognizer2("micr_rec_250313.onnx");
            //micr_dict.txt
        }
        public OcrResult Run(Mat image)
        {
            Size imgSize = image.Size;

            // 보정된 ROI
            var serialRoi = ClampRoi(new Rectangle(700, 120, 160, 48), imgSize);
            var amountRoi = ClampRoi(new Rectangle(230, 250, 320, 48), imgSize);
            var micrRoi = ClampRoi(new Rectangle(0, 20, image.Width, 100), imgSize);


            var serial = rec1.Run(detector.CropAndResize(image, serialRoi, 320, 48));
            var amount = rec1.Run(detector.CropAndResize(image, amountRoi, 320, 48));
            var micr = rec2.Run(new Mat(image, micrRoi).Clone());

            return new OcrResult(serial, amount, micr);
        }

        private Rectangle ClampRoi(Rectangle roi, Size imgSize)
        {
            int x = Math.Max(0, roi.X);
            int y = Math.Max(0, roi.Y);

            int width = roi.Width;
            int height = roi.Height;

            if (x + width > imgSize.Width)
                width = imgSize.Width - x;

            if (y + height > imgSize.Height)
                height = imgSize.Height - y;

            width = Math.Max(1, width);
            height = Math.Max(1, height);

            return new Rectangle(x, y, width, height);
        }
    }
}