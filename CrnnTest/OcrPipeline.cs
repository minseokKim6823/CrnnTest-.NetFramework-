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

            // 비율 기반 ROI 정의 (x,y,width,height)
            var serialRoi = GetRoiByRatio(image, 0.6f, 0.12f, 0.25f, 0.1f);  // 일련번호
            var amountRoi = GetRoiByRatio(image, 0.18f, 0.23f, 0.4f, 0.15f); // 금액
            var micrRoi = GetRoiByRatio(image, 0f, 0.85f, 1f, 0.15f);         // MICR

            var serial = rec1.Run(detector.CropAndResize(image, serialRoi, 320, 48));
            var amount = rec1.Run(detector.CropAndResize(image, amountRoi, 320, 48));
            var micr = rec2.Run(detector.CropAndResize(image, micrRoi, micrRoi.Width, micrRoi.Height));

            // 박스 그리기
            Mat boxImg = image.Clone();
            //CvInvoke.Rectangle(boxImg, serialRoi, new MCvScalar(0, 0, 255), 2);
            //CvInvoke.Rectangle(boxImg, amountRoi, new MCvScalar(0, 255, 0), 2);
            //CvInvoke.Rectangle(boxImg, micrRoi, new MCvScalar(255, 0, 0), 2);

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