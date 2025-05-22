using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
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
        }
        public OcrResult Run(Mat image, List<Rectangle> detectedBoxes)
        {
            Size imgSize = image.Size;

            var serialRoi = GetRoiByRatio(image, 0.75f, 0.15f, 0.05f, 0.05f);
            var amountRoi = GetRoiByRatio(image, 0.27f, 0.25f, 0.3f, 0.05f);
            var micrRoi = GetRoiByRatio(image, 0f, 0.85f, 1f, 0.18f);

            string serial = "", amount = "", micr = "";

            Mat boxImg = image.Clone();

            foreach (var box in detectedBoxes)
            {
                if (box.IntersectsWith(serialRoi) || box.IntersectsWith(amountRoi) || box.IntersectsWith(micrRoi))
                {
                    CvInvoke.Rectangle(boxImg, box, new MCvScalar(255, 0, 0), 2);  // 노란색
                }
                else
                {
                    CvInvoke.Rectangle(boxImg, box, new MCvScalar(128, 128, 128), 1);  // 회색
                }
            }

            List<string> serialParts = new List<string>();
            List<string> amountParts = new List<string>();
            List<string> micrParts = new List<string>();

            foreach (var box in detectedBoxes.OrderBy(b => b.X)) 
            {
                if (box.IntersectsWith(serialRoi))
                {
                    var cropped = detector.CropAndResize(image, box, 1000, 48);
                    serialParts.Add(rec1.Run(cropped));
                    CvInvoke.Imshow("SerialCrop", cropped);
                    CvInvoke.WaitKey(0);
                }
                else if (box.IntersectsWith(amountRoi))
                {
                    var cropped = detector.CropAndResize(image, box, 320, 48);
                    amountParts.Add(rec1.Run(cropped));
                    CvInvoke.Imshow("AmountCrop", cropped);
                    CvInvoke.WaitKey(0);
                }
                else if (box.IntersectsWith(micrRoi))
                {
                    var cropped = detector.CropAndResize(image, box, 320, 48);
                    micrParts.Add(rec2.Run(cropped));
                    CvInvoke.Imshow("MicrCrop", cropped);
                    CvInvoke.WaitKey(0);
                }
            }

            serial = string.Join("   ", serialParts);
            amount = string.Join("   ", amountParts);
            micr = string.Join("   ", micrParts);

            return new OcrResult(serial, amount, micr, boxImg);
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

        private Rectangle GetRoiByRatio(Mat image, float xRatio, float yRatio, float wRatio, float hRatio)
        {
            int x = (int)(xRatio * image.Width);
            int y = (int)(yRatio * image.Height);
            int width = (int)(wRatio * image.Width);
            int height = (int)(hRatio * image.Height);

            return ClampRoi(new Rectangle(x, y, width, height), image.Size);
        }

    }
}