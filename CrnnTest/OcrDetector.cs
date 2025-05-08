using Emgu.CV;
using Emgu.CV.CvEnum;
using System;
using System.Drawing;

namespace CrnnTest
{
    public class OcrDetector
    {
        private string modelPath;

        public OcrDetector(string modelPath)
        {
            this.modelPath = modelPath;
        }

        public Mat CropAndResize(Mat src, Rectangle roi, int width, int height)
        {
            Rectangle safeRoi = ClampRoi(roi, src.Size);
            if (safeRoi.Width <= 0 || safeRoi.Height <= 0)
            {
                // 빈 이미지 반환
                return new Mat(src.Size, DepthType.Cv8U, 3); // 혹은 null 처리
            }

            Mat cropped = new Mat(src, safeRoi);
            Mat resized = new Mat();
            CvInvoke.Resize(cropped, resized, new Size(width, height));
            return resized;
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
