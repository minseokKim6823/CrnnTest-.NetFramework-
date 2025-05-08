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
            //detector = new OcrDetector("ch_PP-OCRv4_det_infer.onnx");
            rec1 = new OcrRecognizer("num_rec_250312.onnx");
            rec2 = new OcrRecognizer2("micr_rec_250313.onnx");
            //micr_dict.txt
        }
        public OcrResult Run(Mat image, List<Rectangle> detectedBoxes)
        {
            Size imgSize = image.Size;

            var serialRoi = GetRoiByRatio(image, 0.6f, 0.12f, 0.25f, 0.1f);
            var amountRoi = GetRoiByRatio(image, 0.18f, 0.21f, 0.4f, 0.1f);
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

            List<string> micrParts = new List<string>();

            foreach (var box in detectedBoxes.OrderBy(b => b.X)) // 왼쪽부터 오른쪽 정렬
            {
                if (box.IntersectsWith(serialRoi))
                {
                    var cropped = detector.CropAndResize(image, box, 320, 48);
                    serial = rec1.Run(cropped);
                }
                else if (box.IntersectsWith(amountRoi))
                {
                    var cropped = detector.CropAndResize(image, box, 320, 48);
                    amount = rec1.Run(cropped);
                }
                else if (box.IntersectsWith(micrRoi))
                {
                    var cropped = detector.CropAndResize(image, box, 320, 48);
                    micrParts.Add(rec2.Run(cropped));
                }
            }

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